"""
Performs a screening for the yearly performance of a LSC-PM with different tilt angles at a given location.
"""

import logging
import time
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pvlib.location import Location
from pvtrace import *

from miniplant.simulation_runner import run_direct_simulation
from miniplant.solar_data import solar_data_for_place_and_time

RAYS_PER_SIMULATIONS = 100

logger = logging.getLogger("pvtrace").getChild("miniplant")


def evaluate_tilt_angle(tilt_angle: int, location: Location, time_range: pd.DatetimeIndex, workers: int = None,
                        simulate_diffuse = False):
    logger.info(f"Starting simulation w/ tilt angle {tilt_angle}")

    solar_data = solar_data_for_place_and_time(location, time_range, tilt_angle)

    def calculate_efficiency_for_datapoint(df):
        """
        This function is apply()ed to the dataframe to populate it with the simulation results.
        It takes care of setting up the simulation, and fill in the relevant fields or it terminates early if the
        simulation is not deemed necessary (solar position below horizon or invalid surface fraction)
        """
        logger.info(f"Current date/time {df.name}")
        # Ensure column existence
        df['simulation_direct'] = 0
        df['direct_reacted'] = 0

        # If spectrum is not valid (close to sunset/sunrise) skip simulation. This is a SPCTRAL2 issue ;)
        if np.count_nonzero(df['direct_spectrum']._y) == 0:
            print(f"skipping this point {df.name} due to low spectrum")
            return df

        # Create a function sampling the current solar spectrum
        def direct_photon_factory() -> float:
            return df['direct_spectrum'].sample(np.random.uniform())

        # Get the fraction of direct photon reacted
        df['simulation_direct'] = run_direct_simulation(tilt_angle=tilt_angle, solar_azimuth=df['azimuth'],
                                                        solar_elevation=df['apparent_elevation'],
                                                        solar_spectrum_function=direct_photon_factory,
                                                        num_photons=RAYS_PER_SIMULATIONS, workers=workers)
        df['direct_reacted'] = df['simulation_direct'] * df['direct_irradiance']

        if simulate_diffuse:
            # Get the fraction of diffuse photon reacted
            df['simulation_direct'] = run_direct_simulation(tilt_angle=tilt_angle, solar_azimuth=df['azimuth'],
                                                            solar_elevation=df['apparent_elevation'],
                                                            solar_spectrum_function=direct_photon_factory,
                                                            num_photons=RAYS_PER_SIMULATIONS, workers=workers)
            df['diffuse_reacted'] = df['simulation_diffuse'] * df['diffuse_irradiance']

        return df

    start_time = time.time()
    results = solar_data.apply(calculate_efficiency_for_datapoint, axis=1)
    print(f"Simulation ended in {(time.time() - start_time) / 60:.1f} minutes!")

    target_file = Path(f"fake_simulation_results/{location.name}/{location.name}_{tilt_angle}deg_results.csv")

    target_file.parent.mkdir(parents=True, exist_ok=True)
    # Saved CSV now include direct_irradiation_simulation_result and dni_reacted! :)
    results.to_csv(target_file, columns=("apparent_elevation", "azimuth",
                                         "simulation_direct", "dni_reacted"))


if __name__ == '__main__':
    # Set loggers
    logging.getLogger('trimesh').disabled = True
    logging.getLogger('shapely.geos').disabled = True
    logging.getLogger("pvtrace").setLevel(logging.INFO)  # use logging.DEBUG for more printouts

    from miniplant.locations import EINDHOVEN

    site = EINDHOVEN

    # Run simulations with the following time range
    one_year_half_hour_resolution = pd.date_range(start=datetime.datetime(2020, 1, 1),
                                                  end=datetime.datetime(2021, 1, 1), freq='0.5H', tz=site.tz)
    tilt_range = [75, 80, 85, 90]

    for tilt in tilt_range:
        evaluate_tilt_angle(tilt_angle=tilt, location=EINDHOVEN, time_range=one_year_half_hour_resolution, workers=4)
