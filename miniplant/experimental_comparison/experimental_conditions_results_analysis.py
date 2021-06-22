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

from miniplant.experimental_comparison.simulate_experimental_conditions import XP_END, XP_START

GOLDEN_RATIO = (1 + 5 ** 0.5) / 2

search_path = Path(".")
result_files = Path("./experimental_conditions_results.csv")

fig, ax = plt.subplots(ncols=1)

# Load data
df = pd.read_csv(result_files, parse_dates=[0], index_col=0, date_parser=lambda col: pd.to_datetime(col, utc=True))
# Resample Hourly
# hourly = df.resample("H").sum()
hourly = df
# Sum direct and diffuse components
hourly["total_reacted"] = hourly["diffuse_reacted"] + hourly["direct_reacted"]

# Plot efficiency
# Set axis label
ax.set_ylabel("Hourly absorbed photon flux (mol/h)")

ax.set_xlim([XP_START, XP_END])

# Date formatter to only show the month from the datetime object, and locator to show every month
date_form = DateFormatter("%H")  #  Show hour of the day
# Location comparison
# ax[ix].set_title(caption+f"TOT: {tot:.0f}")
# Hardcoded titles

maxy = max(hourly["total_reacted"])*1.05
ax.set_ylim(0, maxy)
ax.xaxis.set_major_formatter(date_form)
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))

ax.plot(hourly.index, hourly["total_reacted"], linewidth=0.5)
# Add legend
ax.fill_between(hourly.index, 0, hourly["direct_reacted"], alpha=0.5, label="direct irradiation")
ax.fill_between(hourly.index, hourly["direct_reacted"], hourly["total_reacted"], alpha=0.5, label="diffuse irradiation")
# ax[ix].legend(loc="upper right", ncol=1)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center')

fig.tight_layout()
plt.subplots_adjust(bottom=0.2)  # Space for legend
# plt.show()
plt.savefig("Exp_conditions_simulation.png", dpi=300)
input()
