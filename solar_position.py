import datetime
import pandas as pd
from pvlib.location import Location

# Format: lat/long, timezone, elevation, name
locations = {
    "ein": Location(51.4416, 5.6497, 'Europe/Amsterdam', 17, 'Eindhoven'),
    "tus": Location(32.2, -111, 'US/Arizona', 700, 'Tucson')
}

# Time points to be calculated
times = pd.date_range(start=datetime.datetime(2020, 1, 1), end=datetime.datetime(2020, 12, 31), freq='1H',)

calculation_results = {}
export_cols = ("apparent_elevation", "azimuth", "ghi", "dhi")
for site in locations.values():
    times_ein = times.tz_localize(site.pytz, ambiguous="NaT", nonexistent="NaT")
    # Solar position for given time/place
    ephemeridis = site.get_solarposition(times_ein)
    # Clear sky irradiance for given time/place
    clearsky = site.get_clearsky(times_ein)

    # Merge ephemeridis and clear sky dataframe
    calculation_results[site.name] = pd.concat([ephemeridis, clearsky], axis=1)
    # Filter only hours with sun
    calculation_results[site.name].query('apparent_elevation>0', inplace=True)
    # Export to CSV
    calculation_results[site.name].to_csv(f"Results_{site.name}.csv", columns=export_cols)

