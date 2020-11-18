import datetime

import numpy as np
import pandas as pd
import logging
import time

from pvlib.location import Location

from basic_reactor_simulation import run_simulation
from solar_data import solar_data_for_place_and_time

LOCATION = Location(51.4416, 5.6497, 'Europe/Amsterdam', 17, 'Eindhoven')
TIME_RANGE = pd.date_range(start=datetime.datetime(2020, 1, 1), end=datetime.datetime(2021, 1, 1), freq='2H')


def evaluate_tilt_angle(tilt_angle: int):
    logger = logging.getLogger("pvtrace").getChild("miniplant")
    logger.info(f"Starting simulation w/ tilt angle {tilt_angle}")

    solar_data = solar_data_for_place_and_time(LOCATION, TIME_RANGE)

    def simulate_scene(df) -> float:
        """ Actually perform raytracing simulation w/ pvtrace and returns efficiency """
        # If spectrum is not valid (close to sunset/sunrise) skip simulation.
        if np.count_nonzero(df['spectrum']._y) == 0:
            df['efficiency'] = 0
            df['efficiency_corrected'] = 0
            return df

        def photon_factory() -> float:
            return df['spectrum'].sample(np.random.uniform())

        df['efficiency'] = run_simulation(tilt_angle=tilt_angle, solar_azimuth=df['azimuth'],
                                          solar_elevation=df['apparent_elevation'],
                                          solar_spectrum_function=photon_factory, num_photons=100)
        logger.info(f"Simulation completed for {df.name}")
        df['efficiency_corrected'] = df['efficiency'] * df['ghi']
        return df

    start = time.time()
    results = solar_data.apply(simulate_scene, axis=1)
    print(f"time is {time.time() - start}")

    year_sum = results['efficiency_corrected'].sum()
    print(f"Year sum for this is {year_sum}")

    # These now include efficiency and efficiency_corrected
    results.to_csv(f"raw_results/{LOCATION.name}_{tilt_angle}deg_results.csv",
                   columns=("apparent_elevation", "azimuth", "ghi", "dhi", "efficiency", 'efficiency_corrected'))


if __name__ == '__main__':

    # evaluate_tilt_angle(30)

    # Tilt angles to be tested
    tilt_range = [10, 20, 30, 40, 50, 60]

    for tilt in tilt_range:
        evaluate_tilt_angle(tilt)
