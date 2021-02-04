"""
Analyze the results of the simulation including both direct and diffuse component (i.e. full simulation results folder)
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.ticker as mtick

GOLDEN_RATIO = (1 + 5 ** 0.5) / 2

search_path = Path(".")
result_files = list(search_path.rglob('*.csv'))


fig, ax = plt.subplots(ncols=len(result_files))
maxy = 0  # Shared axis labels across plots

for ix, simulation_results_file in enumerate(result_files):
    # Extract location and angle from file name
    location_name = str(simulation_results_file.parent)
    tilt_angle = simulation_results_file.stem[len(location_name)+1:-11]
    caption = f"{location_name} {tilt_angle}°"

    # Load data
    df = pd.read_csv(simulation_results_file, parse_dates=[0], index_col=0, date_parser=lambda col: pd.to_datetime(col, utc=True))
    # Resample daily
    daily = df.resample("D").sum()
    # Sum direct and diffuse components
    sum_column = daily["diffuse_reacted"] + daily["direct_reacted"]
    daily["total_reacted"] = sum_column

    # Plot efficiency
    # Set axis label
    if ix == 0:
        ax[ix].set_ylabel("Daily absorbed photon flux (mol/day)")

    # Set limits to prevent double January ;)
    ax[ix].set_xlim([datetime(2020, 1, 1), datetime(2020, 12, 31)])

    # Date formatter to only show the month from the datetime object, and locator to show every month
    date_form = DateFormatter("%b")  # %b Show month names %m for month numbers
    ax[ix].set_title(caption)
    if max(daily["total_reacted"])*1.05 > maxy:
        maxy = max(daily["total_reacted"])*1.05
    ax[ix].set_ylim(0, maxy)
    ax[ix].xaxis.set_major_formatter(date_form)
    ax[ix].xaxis.set_major_locator(mdates.MonthLocator())

    ax[ix].plot(daily.index, daily["total_reacted"])
    # Add legend
    ax[ix].fill_between(daily.index, 0, daily["direct_reacted"], alpha=0.5, label="direct irradiation")
    ax[ix].fill_between(daily.index, daily["direct_reacted"], daily["total_reacted"], alpha=0.5, label="diffuse irradiation")
    ax[ix].legend(loc="upper right", ncol=1)

fig.tight_layout()
plt.show()
input()