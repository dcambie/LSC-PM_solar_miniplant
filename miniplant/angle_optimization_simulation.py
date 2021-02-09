"""
Performs a screening for the yearly performance of a LSC-PM with different tilt angles at a given location.
"""

import time
import logging
import datetime
from pathlib import Path
from tqdm import tqdm

import os

# Forcing numpy to single thread results in better multiprocessing performance.
# See pvtrace issue #48
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"


# Set loggers (no output needed for progress bar to work!)
logging.getLogger("trimesh").disabled = True
logging.getLogger("shapely.geos").disabled = True
logging.getLogger("pvtrace").setLevel(logging.WARNING)  # use logging.DEBUG for more printouts

import numpy as np
import pandas as pd

from pvlib.location import Location
from pvtrace import *

from miniplant.simulation_runner import run_direct_simulation
from miniplant.solar_data import solar_data_for_place_and_time
from miniplant.utils import PhotonFactory

RAYS_PER_SIMULATIONS = 100
INCLUDE_DYE = True

logger = logging.getLogger("pvtrace").getChild("miniplant")


def evaluate_tilt_angle(
    tilt_angle: int,
    location: Location,
    workers: int = None,
    time_resolution: int = 1800,
):
    """ Run a simulation with the given tilt angle/location combination and save results as CSV """
    logger.info(f"Starting simulation w/ tilt angle {tilt_angle}")

    solar_data = solar_data_for_place_and_time(
        location, tilt_angle, time_resolution=time_resolution
    )

    def calculate_productivity_for_datapoint(df):
        """
        This function is apply()ed to the dataframe to populate it with the simulation results.
        It takes care of setting up the simulation, and fill in the relevant fields or it terminates early if the
        simulation is not deemed necessary (solar position below horizon or invalid surface fraction)
        """
        logger.info(f"Current date/time {df.name}")
        # Ensure column existence
        df["simulation_direct"] = 0
        df["direct_reacted"] = 0

        # If spectrum is not valid (close to sunset/sunrise) skip simulation. This is a SPCTRAL2 issue ;)
        if np.count_nonzero(df["direct_spectrum"]._y) == 0:
            print(f"skipping this point {df.name} due to low spectrum")
            return df

        # Create a function sampling the current solar spectrum
        direct_photon_factory = PhotonFactory(df["direct_spectrum"])

        # Get the fraction of direct photon reacted
        df["simulation_direct"] = run_direct_simulation(
            tilt_angle=tilt_angle,
            solar_azimuth=df["azimuth"],
            solar_elevation=df["apparent_elevation"],
            solar_spectrum_function=direct_photon_factory,
            num_photons=RAYS_PER_SIMULATIONS,
            workers=workers,
            include_dye=INCLUDE_DYE
        )
        df["direct_reacted"] = df["simulation_direct"] * df["direct_irradiance"]

        return df

    start_time = time.time()
    tqdm.pandas()  # Shows nice progress bar
    results = solar_data.progress_apply(calculate_productivity_for_datapoint, axis=1)
    print(f"Simulation ended in {(time.time() - start_time) / 60:.1f} minutes!")

    prefix = f"simulation_results/{location.name}/{location.name}"
    if not INCLUDE_DYE:
        prefix += "_no_dye"
    target_file = Path(f"{prefix}_{np.abs(tilt_angle)}deg_results.csv")

    target_file.parent.mkdir(parents=True, exist_ok=True)
    # Saved CSV now include direct_irradiation_simulation_result and dni_reacted! :)
    results.to_csv(
        target_file,
        columns=("apparent_elevation", "azimuth", "simulation_direct", "direct_reacted"),
    )


if __name__ == "__main__":
    from miniplant.locations import NORTH_CAPE

    site = NORTH_CAPE

    tilt_range = [35, 40, 45, 50, 55, 60, 65]
    for tilt in tilt_range:
        evaluate_tilt_angle(tilt_angle=tilt, location=site, workers=12)
