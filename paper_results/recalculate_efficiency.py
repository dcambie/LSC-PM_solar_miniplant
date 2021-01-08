import numpy as np
import pandas as pd
from pathlib import Path
from pvtrace.material.utils import spherical_to_cart



def surface_incident(tilt_angle: int = 30, solar_elevation: int = 30, solar_azimuth: int = 180):
    reactor_normal = spherical_to_cart(np.radians(tilt_angle), 0)
    solar_light_normal = spherical_to_cart(np.radians(-solar_elevation + 90), np.radians(-solar_azimuth + 180))
    surface_fraction = np.dot(reactor_normal, solar_light_normal)
    return surface_fraction if surface_fraction > 0 else 0


angles = np.arange(0, 91, 5)  # [0 - 90] every 5 degrees

for angle in angles:
    FILE = Path(f"./Eindhoven_{angle}deg_results.csv")

    # Skip missing files
    if not FILE.exists():
        continue

    # Load data in Pandas dataframe
    df = pd.read_csv(FILE, parse_dates=[0], index_col=0, date_parser=lambda col: pd.to_datetime(col, utc=True))

    # Correction function
    def correct_efficiency(df) -> float:
        correction_factor = surface_incident(tilt_angle=angle, solar_elevation=df['apparent_elevation'], solar_azimuth=df['azimuth'])
        df['efficiency_corrected'] = df['efficiency'] * df['ghi'] * abs(correction_factor)
        return df

    # Apply correction
    df = df.apply(correct_efficiency, axis=1)
    # Save as new file
    df.to_csv(FILE)

    efficiency
