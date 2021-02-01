"""
Runs a full simulation on a year, including both diffuse and direct irradiation.
Output in Einstein absorbed.
"""

import time
import datetime
from pathlib import Path
import logging
import os
# Forcing numpy to single thread results in better multiprocessing performance.
# See pvtrace issue #48
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

from pvlib.location import Location
from pvtrace import *

from miniplant.simulation_runner import run_direct_simulation, run_diffuse_simulation
from miniplant.solar_data import solar_data_for_place_and_time

RAYS_PER_SIMULATIONS = 100
logger = logging.getLogger("pvtrace").getChild("miniplant")


class PhotonFactory:
    """ Create a callable sampling the current solar spectrum """
    def __init__(self, spectrum):
        self.spectrum = spectrum

    def __call__(self, *args, **kwargs):
        return self.spectrum.sample(np.random.uniform())


def correct_productivity_per_reactor_area(df):
    """ All the intensities are normalized over 1 m^2 irradiated.
     Here the correction for the actual reactor irradiated area is performed! """
    raise NotImplementedError


def yearlong_simulation(tilt_angle: int, location: Location, workers: int = None, time_resolution: int = 1800):
    logger.info(f"Starting simulation w/ tilt angle {tilt_angle}")

    solar_data = solar_data_for_place_and_time(location, tilt_angle, time_resolution=time_resolution)

    def calculate_productivity_for_datapoint(df):
        """
        This function is apply()ed to the dataframe to populate it with the simulation results.
        It takes care of setting up the simulation, and fill in the relevant fields or it terminates early if the
        simulation is not deemed necessary (solar position below horizon or invalid surface fraction)
        """
        logger.info(f"Current date/time {df.name}")
        # Ensure column existence
        df['simulation_direct'] = 0
        df['direct_reacted'] = 0

        # Create a function sampling the current solar spectrum
        direct_photon_factory = PhotonFactory(df['direct_spectrum'])
        diffuse_photon_factory = PhotonFactory(df['diffuse_spectrum'])

        # Get the fraction of direct photon reacted
        df['simulation_direct'] = run_direct_simulation(tilt_angle=tilt_angle, solar_azimuth=df['azimuth'],
                                                        solar_elevation=df['apparent_elevation'],
                                                        solar_spectrum_function=direct_photon_factory,
                                                        num_photons=RAYS_PER_SIMULATIONS, workers=workers)
        df['direct_reacted'] = df['simulation_direct'] * df['direct_irradiance']

        # Get the fraction of diffuse photon reacted
        df['simulation_diffuse'] = run_diffuse_simulation(tilt_angle=tilt_angle,
                                                         solar_spectrum_function=diffuse_photon_factory,
                                                         num_photons=RAYS_PER_SIMULATIONS, workers=workers)
        df['diffuse_reacted'] = df['simulation_diffuse'] * df['diffuse_irradiance']

        return df

    start_time = time.time()
    results = solar_data.apply(calculate_productivity_for_datapoint, axis=1)
    final_results = results.apply(correct_productivity_per_reactor_area, axis=1)
    print(f"Simulation ended in {(time.time() - start_time) / 60:.1f} minutes!")

    target_file = Path(f"fake_simulation_results/{location.name}/{location.name}_{tilt_angle}deg_results.csv")

    target_file.parent.mkdir(parents=True, exist_ok=True)
    # Saved CSV now include direct_irradiation_simulation_result and dni_reacted! :)
    final_results.to_csv(target_file, columns=("apparent_elevation", "azimuth",
                                               "simulation_direct", "direct_reacted",
                                               "simulation_diffuse", "diffuse_reacted"))


if __name__ == '__main__':
    # Set loggers
    logging.getLogger('trimesh').disabled = True
    logging.getLogger('shapely.geos').disabled = True
    logging.getLogger("pvtrace").setLevel(logging.INFO)  # use logging.DEBUG for more printouts

    from miniplant.locations import EINDHOVEN
    site = EINDHOVEN

    yearlong_simulation(tilt_angle=40, location=EINDHOVEN, workers=4, time_resolution=3600)
