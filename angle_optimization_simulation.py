import datetime

import numpy as np
import pandas as pd
import logging
import time
import collections

from typing import Callable

from pvlib.location import Location
from pvtrace import *
from pvtrace.material.utils import spherical_to_cart

from solar_data import solar_data_for_place_and_time
from scene_creator import create_standard_scene

RAYS_PER_SIMULATIONS = 128
LOCATION = Location(latitude=51.4416, longitude=5.6497, tz='Europe/Amsterdam', altitude=17, name='Eindhoven')
TIME_RANGE = pd.date_range(start=datetime.datetime(2020, 1, 1), end=datetime.datetime(2020, 1, 2), freq='5H')


def run_simulation(tilt_angle: int = 0, solar_elevation: int = 30, solar_azimuth: int = 180,
                   solar_spectrum_function: Callable = lambda: 555,
                   num_photons: int = 100, render: bool = False) -> float:
    # Create scene with the provided parameters
    scene = create_standard_scene(tilt_angle=tilt_angle, solar_elevation=solar_elevation, solar_azimuth=solar_azimuth,
                                  solar_spectrum_function=solar_spectrum_function)

    logger.info(f"Starting ray-tracing with {num_photons} photons (Render is {render})")

    if render:
        renderer = MeshcatRenderer(wireframe=False, open_browser=False)
        renderer.render(scene)

    # finals = []
    # for ray in scene.emit(num_photons):
    #     steps = photon_tracer.follow(scene, ray)
    #     path, events = zip(*steps)
    #     finals.append(events[-1])
    #     if render:
    #         renderer.add_ray_path(path)

    results = scene.simulate(num_rays=num_photons)
    all_workers_results = [item for sublist in results for item in sublist]
    finals = [photon[-1][1] for photon in all_workers_results]

    count_events = collections.Counter(finals)
    efficiency = count_events[Event.REACT] / num_photons
    logger.info(f"*** SIMULATION ENDED *** (Efficiency was {efficiency:.3f})")
    return efficiency


def surface_incident(tilt_angle: float = 30, solar_elevation: float = 30, solar_azimuth: float = 180) -> float:
    """
    This function is crucial for accurate results.
    All the simulations are performed with a fixed number of photons, but the fraction of solar light received in every
    scene is function of tilt angle and solar position.
    This function calculates the fraction of photon flux actually received by the reactor.
    Negative numbers are obtained when irradiation is from the back, and are therefore discarded as the back of the
    reactor is covered.

    The normalization parameter returned is therefore in the range (0, 1), with 1 representing a solar vector normal to
    the reactor.
    """
    # Get reactor and solar light normals
    reactor_normal = spherical_to_cart(np.radians(tilt_angle), 0)
    solar_light_normal = spherical_to_cart(np.radians(-solar_elevation + 90), np.radians(-solar_azimuth + 180))
    # And calculate their dot products
    surface_fraction = np.dot(reactor_normal, solar_light_normal)
    return surface_fraction if surface_fraction > 0 else 0


def evaluate_tilt_angle(tilt_angle: int):
    logger = logging.getLogger("pvtrace").getChild("miniplant")
    logger.info(f"Starting simulation w/ tilt angle {tilt_angle}")

    solar_data = solar_data_for_place_and_time(LOCATION, TIME_RANGE)

    def calculate_efficiency_for_datapoint(df) -> float:
        """
        This function is apply()ed to the dataframe to populate it with the simulation results.
        It takes care of setting up the simulation, and fill in the relevant fields or it terminates early if the
        simulation is not deemed necessary (solar position below horizon or invalid surface fraction)
        """
        # Ensure column existence
        df['efficiency'] = 0
        df['surface_fraction'] = 0
        df['efficiency_corrected'] = 0

        # If spectrum is not valid (close to sunset/sunrise) skip simulation.
        if np.count_nonzero(df['spectrum']._y) == 0:
            return df

        # Calculate surface fraction and exit if 0
        df['surface_fraction'] = surface_incident(tilt_angle, df['apparent_elevation'], df['azimuth'])
        if df['surface_fraction'] == 0:
            return df

        # Create a function sampling the current solar spectrum
        def photon_factory() -> float:
            return df['spectrum'].sample(np.random.uniform())

        # Efficiency is the raw result of the simulation
        df['efficiency'] = run_simulation(tilt_angle=tilt_angle, solar_azimuth=df['azimuth'],
                                          solar_elevation=df['apparent_elevation'],
                                          solar_spectrum_function=photon_factory, num_photons=RAYS_PER_SIMULATIONS)

        # To be normalized with ghi and surface fraction
        df['efficiency_corrected'] = df['efficiency'] * df['ghi'] * df['surface_fraction']
        return df

    start_time = time.time()
    results = solar_data.apply(calculate_efficiency_for_datapoint, axis=1)
    print(f"Simulation ended in {(time.time() - start_time)/60:.1f} minutes!")

    # These now include efficiency and efficiency_corrected
    results.to_csv(f"delme/{LOCATION.name}_{tilt_angle}deg_results.csv",
                   columns=("apparent_elevation", "azimuth", "ghi", "dhi", "efficiency", "surface_fraction",
                            "efficiency_corrected"))


if __name__ == '__main__':
    # Set loggers
    logging.getLogger('trimesh').disabled = True
    logging.getLogger('shapely.geos').disabled = True
    logging.getLogger("pvtrace").setLevel(logging.DEBUG)
    logger = logging.getLogger("pvtrace").getChild("miniplant")

    # tilt_range = [90, 85, 80, 75, 70, 65, 60, 55, 50, 45]
    tilt_range = [90]
    for tilt in tilt_range:
        evaluate_tilt_angle(tilt)