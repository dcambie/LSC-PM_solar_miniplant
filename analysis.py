import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pvtrace.material.utils import spherical_to_cart


def surface_incident(tilt_angle: int = 30, solar_elevation: int = 30, solar_azimuth: int = 180):
    reactor_normal = spherical_to_cart(np.radians(tilt_angle), 0)
    solar_light_normal = spherical_to_cart(np.radians(-solar_elevation + 90), np.radians(-solar_azimuth + 180))
    surface_fraction = np.dot(reactor_normal, solar_light_normal)
    return surface_fraction


angles = [10, 20, 30, 40, 50, 60]
angle_sum = {}
plt.figure()
for angle in angles:
    FILE = Path(f"./saved_results/yearly_2h_ein/Eindhoven_{angle}deg_results.csv")
    df = pd.read_csv(FILE, parse_dates=[0], index_col=0, date_parser=lambda col: pd.to_datetime(col, utc=True))
    daily = df.resample('D').sum()


    def correct_efficiency(df) -> float:
        """ Actually perform raytracing simulation w/ pvtrace and returns efficiency """
        correction_factor = surface_incident(tilt_angle=angle, solar_elevation=df['apparent_elevation'], solar_azimuth=df['azimuth'])
        print(correction_factor)
        df['efficiency_corrected'] = df['efficiency'] * df['ghi'] * abs(correction_factor)
        return df

    daily = daily.apply(correct_efficiency, axis=1)
    # print(daily)
    plt.plot(daily.index, daily["efficiency_corrected"], label=f"{angle} deg")
    plt.legend(loc="upper left")
    angle_sum[angle] = daily['efficiency_corrected'].sum()
plt.show()
