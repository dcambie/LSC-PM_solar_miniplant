import math
import numpy as np
import pandas as pd
from pathlib import Path

from miniplant.locations import PLATAFORMA_SOLAR_ALMERIA

city = PLATAFORMA_SOLAR_ALMERIA.name
angles = np.arange(0, 91, 5)  # [0 - 90] every 5 degrees

for angle in angles:
    FILE = Path(f"{city}/{city}_{angle}deg_results.csv")

    # Skip missing files
    if not FILE.exists():
        continue

    # Load data in Pandas dataframe
    df = pd.read_csv(
        FILE,
        parse_dates=[0],
        index_col=0,
        date_parser=lambda col: pd.to_datetime(col, utc=True),
    )

    # Correction function
    def correct_efficiency(df) -> float:
        # # Field renames
        # df["direct_irradiation_simulation_result"] = df["efficiency"]
        # del df["efficiency"]
        # del df["efficiency_corrected"]

        # Get DNI from GHI and DHI [GHI = DHI + DNI * Cos(elevation)]
        dni = (df["ghi"] - df["dhi"]) / math.cos(
            math.radians(90 - df["apparent_elevation"])
        )
        df["dni_reacted"] = (
            df["direct_irradiation_simulation_result"] * dni * df["surface_fraction"]
        )

        return df

    # Apply correction
    df = df.apply(correct_efficiency, axis=1)

    # Save as new file
    FILE2 = Path(f"{city}/{city}_{angle}deg_results.csv")
    df.to_csv(FILE2)
    print(f"Angle {angle} corrected!")
