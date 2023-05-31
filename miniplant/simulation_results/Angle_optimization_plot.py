import math
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from miniplant.locations import LOCATIONS

GOLDEN_RATIO = (1 + 5**0.5) / 2

for location in LOCATIONS:
    print(f"Working on {location.name}")
    # Define MatPlotLib figure
    fig, ax = plt.subplots(figsize=plt.figaspect(1 / GOLDEN_RATIO))
    ax.set(
        title=f"Tilt angle impact on LSC-PM performance ({location.name})",
        xlabel="Angle (deg)",
        ylabel="Yearly productivity (normalized)",
        xlim=[-5, 95],
        ylim=[55, 105],
    )
    ax.xaxis.set_major_locator(mtick.MaxNLocator(10))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    x = []
    y = []

    # Iterate files for each angle
    angles = np.arange(0, 91, 5)
    for angle in angles:
        FILE = Path(f"{location.name}/{location.name}_{angle}deg_results.csv")
        if not FILE.exists():
            continue

        # Load simulation results
        df = pd.read_csv(
            FILE,
            parse_dates=[0],
            index_col=0,
            date_parser=lambda col: pd.to_datetime(col, utc=True),
        )

        # Yearly total for this tilt angle
        x.append(angle)
        try:
            y.append(df["dni_reacted"].sum())
        except KeyError:
            y.append(df["direct_reacted"].sum())

    # Skip plot if no data are available
    if len(y) == 0:
        plt.close(fig)
        continue

    # Normalize Y
    y = [100 * float(i) / max(y) for i in y]

    # Plot all
    plt.scatter(x, y)

    # Get max X/Y
    index_max = [i for i, j in enumerate(y) if math.isclose(j, 100)].pop()
    plt.scatter(x[index_max], 100, color="orange")
    ax.annotate(
        f"Optimal angle: {x[index_max]}",
        xy=(x[index_max], 100),
        xycoords="data",
        xytext=(-10, -50),
        textcoords="offset points",
        arrowprops=dict(facecolor="black", arrowstyle="simple"),
        horizontalalignment="center",
        verticalalignment="bottom",
    )

    plt.savefig(f"Angle_optimization_results_{location.name}.png", dpi=300)
#    plt.show()
