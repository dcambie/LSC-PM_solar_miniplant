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

    # Relative Air Mass
    relative_airmass: pd.DataFrame = site.get_airmass(times=local_time, solar_position=sol_pos)
    solar_data = pd.concat([sol_pos, relative_airmass], axis=1)
    # print(solar_data.columns)
    # ['apparent_zenith', 'zenith', 'apparent_elevation', 'elevation',
    #        'azimuth', 'equation_of_time', 'airmass_relative', 'airmass_absolute']

    def calculate_spectrum(df):
        """ Calculate diffuse and direct spectra for every time point at the given location and tilt angle """
        df['aoi'] = irradiance.aoi(surface_tilt=tilt_angle, surface_azimuth=180, solar_zenith=df["apparent_zenith"],
                                   solar_azimuth=df["azimuth"])

        # Use SPCTRAL2 model for irradiance on tilted surface
        solar_spectrum = spectrum.spectrl2(apparent_zenith=df["apparent_zenith"], aoi=df['aoi'],
                                           surface_tilt=tilt_angle, ground_albedo=albedo, surface_pressure=pressure,
                                           relative_airmass=df['airmass_relative'],
                                           precipitable_water=water_vapor_content, ozone=ozone,
                                           aerosol_turbidity_500nm=tau500, dayofyear=df.name.dayofyear)

        """
        PoA (plane of array) values already correct for tilt angle. This means the following:
        
        * Direct component is essentially DNI * pvlib.irradiance.aoi_projection() [that is dot product of solar vector
            and surface normal. If the sun is behind the array the result is negative, we will set that to 0 see below.
        * Diffuse component is the dhi corrected with the Hay & Davies 1980 model for sky diffuse component
        
        NOTE: we are neglecting the ground diffuse component here! (I_tilt = I_beam + I_sky + I_ground)
        """

        # Add solar spectra for diffuse and direct to dataframe (trimmed to UV-VIS) [in particular 10-37 is 360--690 nm]
        try:
            df['direct_spectrum'] = Distribution(solar_spectrum['wavelength'][10:37],
                                                 np.squeeze(solar_spectrum['poa_direct'][10:37]))
            df['diffuse_spectrum'] = Distribution(solar_spectrum['wavelength'][10:37],
                                                  np.squeeze(solar_spectrum['poa_sky_diffuse'][10:37]))
        except ValueError:
            # When the sun is behind the array the sign of AoI dot product becomes negative. We skip those time point ;)
            return df

        # Calculate irradiance from spectral results in the spectral range of interest for simulations (see note above)
        df["direct_irradiance"] = integrate.trapz(df['direct_spectrum']._y, df['direct_spectrum']._x)
        df["diffuse_irradiance"] = integrate.trapz(df['diffuse_spectrum']._y, df['diffuse_spectrum']._x)

        return df

    # Filter date-time point where the sun is above the horizon. Guess why ;)
    solar_data.query('apparent_elevation>0', inplace=True)

    # Calculate spectra (diffuse and direct incident on reactor) per each data-time point
    solar_data = solar_data.apply(calculate_spectrum, axis=1)
    # print(solar_data.columns)

    # Export to CSV
    solar_data.to_csv(f"Full_data_{site.name}.csv", columns=('apparent_zenith', 'zenith', 'apparent_elevation', 'elevation',
       'azimuth', 'airmass_relative', 'aoi', 'direct_irradiance', 'diffuse_irradiance', 'direct_spectrum', 'diffuse_spectrum'))

    logger = logging.getLogger("pvtrace").getChild("miniplant")
    logger.info(f"Generated solar data for {site.name} [lat. {site.latitude}, long. {site.longitude}] in the time range"
                f" {datetime_points.min().isoformat()} -- {datetime_points.max().isoformat()}")

    return solar_data


if __name__ == '__main__':
    from miniplant.locations import NORTH_CAPE
    # Time points to be calculated
    year2020 = pd.date_range(start=datetime.datetime(2020, 1, 1), end=datetime.datetime(2021, 1, 1), freq='0.5H')

    # perform calculations
    test_df = solar_data_for_place_and_time(NORTH_CAPE, year2020, 30)