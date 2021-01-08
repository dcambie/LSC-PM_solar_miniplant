import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

GOLDEN_RATIO = (1 + 5 ** 0.5) / 2

# use Viridis as color cycler to show gradient of angles (individual lines are not distinguishable anyway)
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(1, 0, 20)))

# Set figure settings
fig, ax = plt.subplots(figsize=plt.figaspect(1/GOLDEN_RATIO))
ax.set(title="Tilt angle impact on LSC-PM performance (Eindhoven)",
       xlabel="Month",
       ylabel="Reaction absorbed photons (a.u.)")

# Iterate files for each angle
angles = np.arange(0, 91, 5)
for angle in angles:
    # Target file
    FILE = Path(f"Eindhoven_{angle}deg_results.csv")
    if not FILE.exists():
        continue

    # Load simulation results
    df = pd.read_csv(FILE, parse_dates=[0], index_col=0, date_parser=lambda col: pd.to_datetime(col, utc=True))
    # Resample daily
    daily = df.resample('D').sum()

    # Plot efficiency
    plt.plot(daily.index, daily["efficiency_corrected"], label=f"{angle} deg")
    # Add legend
    plt.legend(loc="upper right")

# Date formatter to only show the month from the datetime object, and locator to show every month
date_form = DateFormatter("%m")
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

# Save image
plt.savefig("Yearly_results_Eindhoven_vs_tilt_angle.png", dpi=300)
plt.show()
