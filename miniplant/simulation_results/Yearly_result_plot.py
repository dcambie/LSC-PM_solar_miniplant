import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.ticker as mtick
from miniplant.locations import LOCATIONS

GOLDEN_RATIO = (1 + 5 ** 0.5) / 2

for location in LOCATIONS:
    # use Viridis as color cycler to show gradient of angles (individual lines are not distinguishable anyway)
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(1, 0, 20)))

    # Set figure settings
    fig, ax = plt.subplots(figsize=plt.figaspect(1/GOLDEN_RATIO))
    ax.set(title=f"Tilt angle impact on LSC-PM performance ({location.name})",
           xlabel="Month",
           ylabel="Reaction absorbed photons (a.u.)")
    df = None

    # Iterate files for each angle
    angles = np.arange(0, 91, 5)
    for angle in angles:
        # Target file
        FILE = Path(f"{location.name}/{location.name}_{angle}deg_results.csv")
        if not FILE.exists():
            continue

        # Load simulation results
        df = pd.read_csv(FILE, parse_dates=[0], index_col=0, date_parser=lambda col: pd.to_datetime(col, utc=True))
        # Resample daily
        daily = df.resample('D').sum()

        # Plot efficiency
        plt.plot(daily.index, daily["dni_reacted"], label=f"{angle} deg")
        # Add legend
        plt.legend(loc="upper right")

    # Skip plot if no data are available
    if df is None:
        plt.close(fig)
        continue

    # Date formatter to only show the month from the datetime object, and locator to show every month
    date_form = DateFormatter("%m")
    ax.xaxis.set_major_formatter(date_form)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    # Save image
    plt.savefig(f"Yearly_results_{location.name}_vs_tilt_angle.png", dpi=300)
    plt.show()
