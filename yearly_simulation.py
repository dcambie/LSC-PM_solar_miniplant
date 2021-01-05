import datetime

import numpy as np
import pandas as pd
import logging
import time

from pvlib.location import Location
from pvtrace.material.utils import spherical_to_cart

from basic_reactor_simulation import run_simulation
from solar_data import solar_data_for_place_and_time

LOCATION = Location(latitude=51.4416, longitude=5.6497, tz='Europe/Amsterdam', altitude=17, name='Eindhoven')
TIME_RANGE = pd.date_range(start=datetime.datetime(2020, 1, 1), end=datetime.datetime(2021, 1, 1), freq='0.5H')


def surface_incident(tilt_angle: int = 30, solar_elevation: int = 30, solar_azimuth: int = 180):
    reactor_normal = spherical_to_cart(np.radians(tilt_angle), 0)
    solar_light_normal = spherical_to_cart(np.radians(-solar_elevation + 90), np.radians(-solar_azimuth + 180))
    surface_fraction = np.dot(reactor_normal, solar_light_normal)
    return surface_fraction


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
        df['surface_fraction'] = surface_incident(tilt_angle, df['apparent_elevation'], df['azimuth'])
        df['efficiency_corrected'] = df['efficiency'] * df['ghi'] * df['surface_fraction']
        return df

    start = time.time()
    results = solar_data.apply(simulate_scene, axis=1)
    print(f"time is {time.time() - start}")

    year_sum = results['efficiency_corrected'].sum()
    print(f"Year sum for this is {year_sum}")

    # These now include efficiency and efficiency_corrected
    results.to_csv(f"paper_results/{LOCATION.name}_{tilt_angle}deg_results.csv",
                   columns=("apparent_elevation", "azimuth", "ghi", "dhi", "efficiency", 'efficiency_corrected'))


if __name__ == '__main__':
    # Tilt angles to be tested
    tilt_range = [60]
    import time
    start_time = time.time()
    for tilt in tilt_range:
        evaluate_tilt_angle(tilt)
    print(f"It took {time.time()-start_time:.1f} seconds")
