"""
Sun path diagram Eindhoven
==========================
"""

# From https://pvlib-python.readthedocs.io/en/stable/auto_examples/plot_sunpath_diagrams.html

# This is a cartesian plot of hourly solar zenith and azimuth. The figure-8
# patterns are called `analemmas <https://en.wikipedia.org/wiki/Analemma>`_ and
# show how the sun's path slowly shifts over the course of the year .  The
# colored lines show the single-day sun paths for the winter and summer
# solstices as well as the spring equinox.
#
# The solstice paths mark the boundary of the sky area that the sun traverses
# over a year.  The diagram shows that there is no point in the
# year when is the sun directly overhead (zenith=0) -- note that this location
# is north of the Tropic of Cancer.
#
# Examining the sun path for the summer solstice in particular shows that
# the sun rises north of east, crosses into the southern sky around 10 AM for a
# few hours before crossing back into the northern sky around 3 PM and setting
# north of west.  In contrast, the winter solstice sun path remains in the
# southern sky the entire day.  Moreover, the diagram shows that the winter
# solstice is a shorter day than the summer solstice -- in December, the sun
# rises after 7 AM and sets before 6 PM, whereas in June the sun is up before
# 6 AM and sets after 7 PM.
#
# Another use of this diagram is to determine what times of year the sun is
# blocked by obstacles. For instance, for a mountain range on the western side
# of an array that extends 10 degrees above the horizon, the sun is blocked:
#
# - after about 6:30 PM on the summer solstice
# - after about 5:30 PM on the spring equinox
# - after about 4:30 PM on the winter solstice
from pathlib import Path

from pvlib import solarposition
from pvlib.location import Location
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from miniplant.locations import LOCATIONS

for location in LOCATIONS:
    times = pd.date_range('2019-01-01 00:00:00', '2020-01-01', closed='left', freq='H', tz=location.pytz)
    solpos = solarposition.get_solarposition(time=times, latitude=location.latitude, longitude=location.longitude)

    # remove nighttime
    solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]

    fig, ax = plt.subplots()
    ax.set(xlim=[0, 360])
    ax.xaxis.set_major_locator(mtick.MaxNLocator(6))

    points = ax.scatter(solpos.azimuth, solpos.apparent_elevation, s=2, c=solpos.index.dayofyear, label=None)

    # Color bar
    cbar = fig.colorbar(points, orientation="vertical")
    cbar.ax.get_yaxis().labelpad = 15

    cbar.set_label('Day of the year', rotation=270)

    for hour in np.unique(solpos.index.hour):
        # choose label position by the largest elevation for each hour
        subset = solpos.loc[solpos.index.hour == hour, :]
        height = subset.apparent_elevation
        pos = solpos.loc[height.idxmax(), :]
        ax.text(pos['azimuth'], pos['apparent_elevation'], str(hour))

    for date in pd.to_datetime(['2019-03-21', '2019-06-21', '2019-12-21']):
        times = pd.date_range(date, date+pd.Timedelta('24h'), freq='5min', tz=location.pytz)
        solpos = solarposition.get_solarposition(times, location.latitude, location.longitude)
        # Ignore points where apparent elevation is below horizon ;)
        solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]

        # Export as csv for both use and validation
        export_file_name = f"{location.name}_{date.strftime('%Y-%m-%d')}_solar_position.csv"
        solpos.to_csv(export_file_name, columns=["azimuth", "apparent_elevation"])

        label = date.strftime('%d-%b')
        ax.plot(solpos.azimuth, solpos.apparent_elevation, label=label)

    ax.figure.legend(loc='lower right')
    plt.title(f'Sunpath {location.name} (calculated with pvlib)')
    ax.set_xlabel('Solar Azimuth (degrees)')
    ax.set_ylabel('Solar Elevation (degrees)')

    # Export as figure
    plt.savefig(f"{location.name}_solar_positions.png", dpi=300)

    # And show it as well
    plt.show()
