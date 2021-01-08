import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

GOLDEN_RATIO = (1 + 5 ** 0.5) / 2
fig, ax = plt.subplots(figsize=plt.figaspect(1/GOLDEN_RATIO))
ax.set(title="Tilt angle impact on LSC-PM performance (Eindhoven)",
       xlabel="Angle (deg)",
       ylabel="Yearly productivity (normalized)",
       xlim=[-5, 95],
       ylim=[55, 105])
ax.xaxis.set_major_locator(mtick.MaxNLocator(10))
ax.yaxis.set_major_formatter(mtick.PercentFormatter())


x = []
y = []

# Iterate files for each angle
angles = np.arange(0, 91, 5)
for angle in angles:
    FILE = Path(f"Eindhoven_{angle}deg_results.csv")
    if not FILE.exists():
        continue

    # Load simulation results
    df = pd.read_csv(FILE, parse_dates=[0], index_col=0, date_parser=lambda col: pd.to_datetime(col, utc=True))

    # Yearly total for this tilt angle
    x.append(angle)
    y.append(df['efficiency_corrected'].sum())

# Normalize Y
y = [100*float(i)/max(y) for i in y]

# Plot all
plt.scatter(x, y)

# Get max X/Y
index_max = [i for i, j in enumerate(y) if j == 100].pop()
plt.scatter(x[index_max], 100, color="orange")
ax.annotate(f"Optimal angle: {x[index_max]}", xy=(x[index_max], 100),
            xycoords='data',
            xytext=(-10, -50), textcoords='offset points',
            arrowprops=dict(facecolor='black', arrowstyle='simple'),
            horizontalalignment='center', verticalalignment='bottom')

plt.savefig("Angle_optimization_results_Eindhoven.png", dpi=300)
plt.show()
