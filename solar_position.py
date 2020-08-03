import datetime
import pandas as pd
from pvlib.location import Location

locations = {
    "ein": Location(51.4416, 5.6497, 'Europe/Amsterdam', 17, 'Eindhoven'),
    "tus": Location(32.2, -111, 'US/Arizona', 700, 'Tucson')
}

times = pd.date_range(start=datetime.datetime(2020, 1, 1), end=datetime.datetime(2020, 12, 31), freq='1H',)

calculation_results = {}
export_cols = ("apparent_elevation", "azimuth", "ghi", "dhi")
for site in locations.values():
    times_ein = times.tz_localize(site.pytz, ambiguous="NaT", nonexistent="NaT")
    ephemeridis = site.get_solarposition(times_ein)
    clearsky = site.get_clearsky(times_ein)

    calculation_results[site.name] = pd.concat([ephemeridis, clearsky], axis=1)
    calculation_results[site.name].query('apparent_elevation>0', inplace=True)
    calculation_results[site.name].to_csv(f"Results_{site.name}.csv", columns=export_cols)

