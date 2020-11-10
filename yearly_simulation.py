import pandas as pd
import logging
import time
from basic_reactor_simulation import run_simulation

tilt_range = [10, 20, 30, 40, 50, 60]

for tilt in tilt_range:
    bigsum = 0
    solar_positions = pd.read_csv("Results_Eindhoven_jan.csv")

    LOGGER = logging.getLogger("pvtrace").getChild("miniplant")
    LOGGER.info(f"Starting simulation w/ tilt angle {tilt}")

    def calculate_efficiency(df):
        global bigsum
        df['efficiency'] = run_simulation(tilt_angle=tilt, solar_azimuth=df['azimuth'],
                                          solar_elevation=df['apparent_elevation'])
        df['efficiency_corrected'] = df['efficiency'] * df['ghi']
        bigsum += df['efficiency_corrected']
        return df

    start = time.time()
    results = solar_positions.apply(calculate_efficiency, axis=1)
    results.to_csv(f"raw_results/jan_{tilt}deg_results.csv")
    print(f"{tilt} degree SUM is {bigsum}")
