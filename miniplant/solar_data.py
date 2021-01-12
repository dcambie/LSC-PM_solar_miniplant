import datetime
import logging
import warnings

import pandas as pd

from pvtrace import Distribution
from pvlib.location import Location
from solcore.light_source import calculate_spectrum_spectral2, get_default_spectral2_object


def solar_data_for_place_and_time(site: Location, datetime_points=pd.core.indexes.datetimes.DatetimeIndex) -> pd.core.frame.DataFrame:
    """
    Given a Location object and a series of datetime points calculates relevant solar position and spectral distribution

    :param site: pvlib.location.Location object
    :param datetime_points: as pandas data_range
    :return: a pd.DataFrame with all the relevant results
    """

    # Localize time if timezone is available
    local_time = datetime_points.tz_localize(site.pytz, ambiguous="NaT", nonexistent="NaT")

    # Solar position
    ephemeridis = site.get_solarposition(local_time)

    # Clear sky irradiance
    clearsky = site.get_clearsky(local_time)

    # Spectral distribution
    spectra = []

    # get default stateObject for spectral2 calculation and customize it
    spectral2_input = get_default_spectral2_object()
    spectral2_input["latitude"] = site.latitude
    spectral2_input["longitude"] = site.longitude

    for date_and_time in datetime_points:
        spectral2_input["dateAndTime"] = date_and_time.to_pydatetime()

        # datetime close to sunrise/sunset give invalid spectra. Catch warning related to that
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wavelength, intensity = calculate_spectrum_spectral2(stateObject=spectral2_input, power_density_in_nm=True)
            # Add spectrum to list (trimmed to UV-VIS) [in particular 10-37 is 360--690 nm]
            spectra.append(Distribution(wavelength[10:37], intensity[10:37]))
    spectra = pd.DataFrame(data=spectra, columns=["spectrum"], index=local_time)

    # Merge ephemeridis, clear sky and solar spectra into a single dataframe
    calculation_results = pd.concat([ephemeridis, clearsky, spectra], axis=1)
    # Filter points with sun above horizon
    calculation_results.query('apparent_elevation>0', inplace=True)
    # Export to CSV
    # calculation_results.to_csv(f"Full_data_{site.name}.csv", columns=("apparent_elevation", "azimuth", "ghi", "dhi"))

    logger = logging.getLogger("pvtrace").getChild("miniplant")
    logger.info(f"Generated solar data for {site.name} [lat. {site.latitude}, long. {site.longitude}] in the time range"
                f" {datetime_points.min().isoformat()} -- {datetime_points.max().isoformat()}")

    return calculation_results


if __name__ == '__main__':
    from miniplant.locations import EINDHOVEN
    # Time points to be calculated
    year2020 = pd.date_range(start=datetime.datetime(2020, 1, 1), end=datetime.datetime(2021, 1, 1), freq='1H')

    # perform calculations
    test_df = solar_data_for_place_and_time(EINDHOVEN, year2020)

    print(test_df)  # [4449 rows x 10 columns]
