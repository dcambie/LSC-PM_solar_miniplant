"""
Module with a function to convert spectra from W * m^-2 * nm^-1 to umol * m^-2 * s^-1
"""
from typing import Iterator

import numpy as np
from scipy.constants import Planck, speed_of_light, Avogadro
from pvtrace import Distribution, Ray, Light


def photon_energy(wavelength):
    """ Returns the energy (in J) of a photon of given wavelength in nm """
    return Planck * speed_of_light / (wavelength * 1e-9)


def irradiance_to_photon_flux(irradiance_per_nm, at_wavelength):
    """ Converts W / m^2 into moles / m^2 * s """
    photons = irradiance_per_nm / photon_energy(at_wavelength)
    moles = photons / Avogadro
    return moles


def spectral_distribution_to_photon_distribution(distribution: Distribution, integration_time=1800):
    """ Given a pvtrace Distribution in W/m^2 converts it into photon flux """
    photon_flux = []

    for wavelength, intensity in zip(distribution._x, distribution._y):
        photon_flux.append(irradiance_to_photon_flux(intensity, wavelength)* integration_time)
    return Distribution(distribution._x, np.array(photon_flux))


if __name__ == '__main__':
    import math

    assert math.isclose(photon_energy(300), 6.621486190496428e-19)
    assert math.isclose(irradiance_to_photon_flux(1, 300), 2.5078041687335337e-06)

    wl = np.array([300, 305])
    i = np.array([1, 2])
    spectral = Distribution(wl, i)
    photon = spectral_distribution_to_photon_distribution(spectral)
    print(photon)  # <pvtrace.material.distribution.Distribution object at 0x0000024B817A02E0>
    print(photon._y)  # [0.00451405 0.00917856]


class PhotonFactory:
    """ Create a callable sampling the current solar spectrum """
    def __init__(self, spectrum):
        self.spectrum = spectrum

    def __call__(self, *args, **kwargs):
        return self.spectrum.sample(np.random.uniform())


class MyLight(Light):
    """ Modified pvtracel.Light object """

    def __init__(self, wavelength=None, position_and_direction=None, name="Light"):
        self.wavelength = wavelength
        self.position_direction = position_and_direction
        self.name = name

    def emit(self, num_rays=None) -> Iterator[Ray]:
        if num_rays is None or num_rays == 0:
            return
        count = 0
        while True:
            count += 1
            if num_rays is not None and count > num_rays:
                break

            try:
                position, direction = self.position_direction()
                ray = Ray(
                    wavelength=self.wavelength(),
                    position=position,
                    direction=direction,
                    source=self.name,
                )
            except Exception:
                raise
            yield ray
