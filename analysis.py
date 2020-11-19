import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

angles = [10, 20, 30, 40, 50, 60]
angle_sum = {}
plt.figure()
for angle in angles:
    FILE = Path(f"./raw_results/Eindhoven_{angle}deg_results.csv")
    df = pd.read_csv(FILE, parse_dates=[0], index_col=0, date_parser=lambda col: pd.to_datetime(col, utc=True))
    daily = df.resample('D').sum()
    # plt.plot(daily.index, daily["efficiency_corrected"])
    plt.plot(daily.index, daily["efficiency_corrected"])
    angle_sum[angle] = daily['efficiency_corrected'].sum()

plt.show()
print(angle_sum)
plt.figure()
plt.plot(angle_sum.keys(), angle_sum.values())
plt.show()
