"""
Performs a screening for the yearly performance of a LSC-PM with different tilt angles at a given location.
"""

import datetime
import math
from pathlib import Path

import os
# Forcing numpy to single thread results in better multiprocessing performance.
# See pvtrace issue #48
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import logging
import time
import collections

from typing import Callable

from pvlib.location import Location
from pvtrace import *
from pvtrace.material.utils import spherical_to_cart

from miniplant.solar_data import solar_data_for_place_and_time
from miniplant.scene_creator import create_standard_scene

RAYS_PER_SIMULATIONS = 100

# Yes, these are arbitrarily chosen as examples of different latitude. Sorry for Europe-centrism ;)

# Run simulations with the following time range
TIME_RANGE = pd.date_range(start=datetime.datetime(2020, 1, 1), end=datetime.datetime(2021, 1, 1), freq='0.5H')
# Test script with this
# TIME_RANGE = pd.date_range(start=datetime.datetime(2020, 1, 1), end=datetime.datetime(2020, 1, 2), freq='5H')

logger = logging.getLogger("pvtrace").getChild("miniplant")


def run_simulation(tilt_angle: int = 0, solar_elevation: int = 30, solar_azimuth: int = 180,
                   solar_spectrum_function: Callable = lambda: 555,
                   num_photons: int = 100, render: bool = False, workers: int = None) -> float:
    # Create scene with the provided parameters
    scene = create_standard_scene(tilt_angle=tilt_angle, solar_elevation=solar_elevation, solar_azimuth=solar_azimuth,
                                  solar_spectrum_function=solar_spectrum_function)

    logger.debug(f"Starting ray-tracing with {num_photons} photons (Render is {render})")

    if render:
        renderer = MeshcatRenderer(wireframe=False, open_browser=False)
        renderer.render(scene)

    # SINGLE-THREADED
    # finals = []
    # for ray in scene.emit(num_photons):
    #     steps = photon_tracer.follow(scene, ray)
    #     path, events = zip(*steps)
    #     finals.append(events[-1])
    #     if render:
    #         renderer.add_ray_path(path)

    # MULTI-THREADED
    results = scene.simulate(num_rays=num_photons, workers=workers)
    finals = [photon[-1][1] for photon in results]

    count_events = collections.Counter(finals)
    efficiency = count_events[Event.REACT] / num_photons
    logger.debug(f"*** SIMULATION ENDED *** (Efficiency was {efficiency:.3f})")
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
    return surface_fraction


class PhotonFactory:
    """ Create a callable sampling the current solar spectrum """
    def __init__(self, spectrum):
        self.spectrum = spectrum

    def __call__(self, *args, **kwargs):
        return self.spectrum.sample(np.random.uniform())


def evaluate_tilt_angle(tilt_angle: int, location: Location, workers: int = None):
    logger.info(f"Starting simulation w/ tilt angle {tilt_angle}")

    solar_data = solar_data_for_place_and_time(location, TIME_RANGE)

    def calculate_efficiency_for_datapoint(df) -> float:
        """
        This function is apply()ed to the dataframe to populate it with the simulation results.
        It takes care of setting up the simulation, and fill in the relevant fields or it terminates early if the
        simulation is not deemed necessary (solar position below horizon or invalid surface fraction)
        """
        logger.info(f"Current date/time {df.name}")
        # Ensure column existence
        df['direct_irradiation_simulation_result'] = 0
        df['surface_fraction'] = 0
        df['dni_reacted'] = 0

        # If spectrum is not valid (close to sunset/sunrise) skip simulation. This is a SPCTRAL2 issue ;)
        if np.count_nonzero(df['spectrum']._y) == 0:
            return df

        # Calculate surface fraction and exit if <0, i.e. if irradiation is coming from the back of the reactor
        df['surface_fraction'] = surface_incident(tilt_angle, df['apparent_elevation'], df['azimuth'])
        if df['surface_fraction'] < 0:
            return df

        # Create a function sampling the current solar spectrum
        photon_factory = PhotonFactory(df['spectrum'])

        # Efficiency is the raw result of the simulation
        df['direct_irradiation_simulation_result'] = run_simulation(tilt_angle=tilt_angle, solar_azimuth=df['azimuth'],
                                                                    solar_elevation=df['apparent_elevation'],
                                                                    solar_spectrum_function=photon_factory,
                                                                    num_photons=RAYS_PER_SIMULATIONS, workers=workers)

        # To be normalized with ghi and surface fraction
        dni = (df['ghi'] - df['dhi']) / math.cos(math.radians(df['apparent_elevation']))
        # Correct for the fraction of reactor surface projected on the normal to the solar vector (W*m-2)
        df['dni_reacted'] = df['direct_irradiation_simulation_result'] * dni * df['surface_fraction']

        return df

    start_time = time.time()
    results = solar_data.apply(calculate_efficiency_for_datapoint, axis=1)
    print(f"Simulation ended in {(time.time() - start_time) / 60:.1f} minutes!")

    target_file = Path(f"simulation_results/{location.name}/{location.name}_{np.abs(tilt_angle)}deg_results.csv")
    target_file.parent.mkdir(exist_ok=True)
    # Saved CSV now include direct_irradiation_simulation_result and dni_reacted! :)
    results.to_csv(target_file, columns=("apparent_elevation", "azimuth", "ghi", "dhi", "surface_fraction",
                                         "direct_irradiation_simulation_result", "dni_reacted"))


if __name__ == '__main__':
    # Set loggers
    logging.getLogger('trimesh').disabled = True
    logging.getLogger('shapely.geos').disabled = True
    logging.getLogger("pvtrace").setLevel(logging.INFO)  # use logging.DEBUG for more printouts

    from miniplant.locations import EINDHOVEN
    tilt_range = [75, 80, 85, 90]
    for tilt in tilt_range:
        evaluate_tilt_angle(tilt, EINDHOVEN, workers=4)
