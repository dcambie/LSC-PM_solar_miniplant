""""
Calculate solar irradiance and spectrum incident on reactor at given position/time/tilt
"""
import datetime
import logging

import pandas as pd
import numpy as np
import scipy.integrate as integrate

import matplotlib.pyplot as plt


from pvtrace import Distribution
from pvlib.location import Location
from pvlib import spectrum, solarposition, irradiance, atmosphere
# from solcore.light_source import calculate_spectrum_spectral2, get_default_spectral2_object
import solcore.light_source.smarts

# assumptions
water_vapor_content = 0.5  # cm
tau500 = 0.1
ozone = 0.31  # atm-cm
albedo = 0.2


def solar_data_for_place_and_time(site: Location, datetime_points: pd.core.indexes.datetimes.DatetimeIndex,
                                  tilt_angle: int) -> pd.core.frame.DataFrame:
    """
    Given a Location object and a series of datetime points calculates relevant solar position and spectral distribution

    :param site: pvlib.location.Location object
    :param datetime_points: as pandas data_range
    :param tilt_angle: reactor tilt angle, used to calculate angle of incidence
    :return: a pd.DataFrame with all the relevant results
    """
    # Pressure based on site altitude
    pressure = atmosphere.alt2pres(site.altitude)

    # Localize time if timezone is available
    local_time = datetime_points.tz_localize(site.pytz, ambiguous="NaT", nonexistent="NaT")

    # Solar position
    sol_pos: pd.DataFrame = site.get_solarposition(times=local_time)

    # Clear sky irradiance replaced with integral over spctral2 ROI
    clearsky_irradiance = site.get_clearsky(times=local_time, solar_position=sol_pos)

    # Relative Air Mass
    relative_airmass: pd.DataFrame = site.get_airmass(times=local_time, solar_position=sol_pos)
    solar_data = pd.concat([sol_pos, relative_airmass, clearsky_irradiance], axis=1)
    # print(solar_data.columns)
    # ['apparent_zenith', 'zenith', 'apparent_elevation', 'elevation',
    #        'azimuth', 'equation_of_time', 'airmass_relative', 'airmass_absolute',
    #        'ghi', 'dni', 'dhi']

    def calculate_spectrum(df):
        """ Calculate diffuse and direct spectra for every time point at the given location and tilt angle """
        df['aoi'] = irradiance.aoi(surface_tilt=tilt_angle, surface_azimuth=180, solar_zenith=df["apparent_zenith"],
                                   solar_azimuth=df["azimuth"])

        solar_spectrum = spectrum.spectrl2(apparent_zenith=df["apparent_zenith"], aoi=df['aoi'],
                                           surface_tilt=tilt_angle, ground_albedo=albedo, surface_pressure=pressure,
                                           relative_airmass=df['airmass_relative'],
                                           precipitable_water=water_vapor_content, ozone=ozone,
                                           aerosol_turbidity_500nm=tau500, dayofyear=df.name.dayofyear)

        # Add spectra to dataframe (trimmed to UV-VIS) [in particular 10-37 is 360--690 nm]
        df['direct_spectrum'] = Distribution(solar_spectrum['wavelength'][10:37], solar_spectrum['dni'][10:37])
        df['diffuse_spectrum'] = Distribution(solar_spectrum['wavelength'][10:37],
                                              solar_spectrum['poa_sky_diffuse'][10:37])

        x = solar_spectrum['wavelength']
        y = np.squeeze(solar_spectrum['dni'])
        y2 = np.squeeze(solar_spectrum['dhi'])
        df["dni_spctral"] = integrate.trapz(y, x)
        df["dhi_spctral"] = integrate.trapz(y2, x)

        return df

    # Filter date-time point where the sun is above the horizon. Guess why ;)
    solar_data.query('apparent_elevation>0', inplace=True)

    # Calculate spectra (diffuse and direct incident on reactor) per each data-time point
    solar_data = solar_data.apply(calculate_spectrum, axis=1)
    print(solar_data.columns)


    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.scatter(solar_data['dni'], solar_data['dni_spctral'])
    ax2.scatter(solar_data['dhi'], solar_data['dhi_spctral'])
    plt.show()
    plt.savefig("comparison_spectral_get_clearsky.png")

    # Export to CSV
    solar_data.to_csv(f"Full_data_{site.name}.csv", columns=('apparent_zenith', 'zenith', 'apparent_elevation', 'elevation',
       'azimuth', 'airmass_relative', 'aoi', 'dni', 'dhi', 'dni_spctral', 'dhi_spctral'))

    logger = logging.getLogger("pvtrace").getChild("miniplant")
    logger.info(f"Generated solar data for {site.name} [lat. {site.latitude}, long. {site.longitude}] in the time range"
                f" {datetime_points.min().isoformat()} -- {datetime_points.max().isoformat()}")

    return solar_data


if __name__ == '__main__':
    from miniplant.locations import NORTH_CAPE
    # Time points to be calculated
    year2020 = pd.date_range(start=datetime.datetime(2020, 3, 30), end=datetime.datetime(2020, 4, 1), freq='0.5H')

    # perform calculations
    test_df = solar_data_for_place_and_time(NORTH_CAPE, year2020, 30)

    print(test_df)

