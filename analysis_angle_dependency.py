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


angles = np.arange(0, 91, 5)
angles = [45, 70, 75, 80, 85, 90]
angle_sum = {}
plt.figure()
for angle in angles:
    FILE = Path(f"./paper_results/Eindhoven_{angle}deg_results.csv")
    df = pd.read_csv(FILE, parse_dates=[0], index_col=0, date_parser=lambda col: pd.to_datetime(col, utc=True))

    daily = df.resample('D').sum()
    plt.plot(daily.index, daily["efficiency_corrected"], label=f"{angle} deg")

    # weekly = df.resample('D').sum()
    # plt.plot(weekly.index, weekly["efficiency_corrected"], label=f"{angle} deg")

    # monthly = df.resample('M').sum()
    # plt.plot(monthly.index, monthly["efficiency_corrected"], label=f"{angle} deg")

    plt.legend(loc="upper left")
    angle_sum[angle] = daily['efficiency_corrected'].sum()
plt.show()
plt.clf()


factor = max(angle_sum.values())
x = []
y = []
for k in angle_sum:
    x.append(k)
    y.append(angle_sum[k] / factor)

print(x)
print(y)
plt.plot(x, y)
plt.show()
