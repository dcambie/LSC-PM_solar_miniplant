"""
Module with a function to convert spectra from W * m^-2 * nm^-1 to umol * m^-2 * s^-1
"""
from typing import Iterator

import numpy as np
from pvlib import irradiance
from pvtrace.geometry.transformations import rotation_matrix
from pvtrace.material.utils import spherical_to_cart
from scipy.constants import Planck, speed_of_light, Avogadro
from pvtrace import Distribution, Ray, Light, rectangular_mask


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
        photon_flux.append(irradiance_to_photon_flux(intensity, wavelength) * integration_time)
    return Distribution(distribution._x, np.array(photon_flux))


class PhotonFactory:
    """ Create a callable sampling the current solar spectrum """

    def __init__(self, spectrum):
        self.spectrum = spectrum

    def __call__(self, *args, **kwargs):
        return self.spectrum.sample(np.random.uniform())


class MyLight(Light):
    """ Modified pvtrace.Light object """

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


class VectorInverter:
    def __init__(self, vector):
        self.vector = vector

    def __call__(self, *args, **kwargs):
        return tuple(-value for value in self.vector)


class LightPosition:
    def __init__(self, tilt_angle):
        self.tilt_angle = tilt_angle
        pass

    def __call__(self, *args, **kwargs):
        position = rectangular_mask(0.47 / 2, 0.47 / 2)
        matrix = np.linalg.inv(rotation_matrix(np.radians(-self.tilt_angle), (0, 1, 0)))

        homogeneous_pt = np.ones(4)
        homogeneous_pt[0:3] = position
        new_pt = np.dot(matrix, homogeneous_pt)[0:3]
        return tuple(new_pt)


def create_diffuse_photon(tilt_angle: int = 30) -> np.ndarray:
    # Keep on generating random photons until they hit the front face of the reactor (not the back)
    # This is correct because poa_diffuse already takes into account the tilt angle! ;)

    # Angle of Incidence projection is positive for front face and negative for back
    aoi_projection = -1
    while aoi_projection < 0:
        # Get a new random point in the half-sphere
        random_azimuth = np.random.rand() * 360
        random_zenith = np.random.rand() * 90
        # Test its validity
        aoi_projection = irradiance.aoi_projection(
            surface_tilt=tilt_angle,
            surface_azimuth=0,
            solar_zenith=random_zenith,
            solar_azimuth=random_azimuth,
        )
    # Return the corresponding vector (note that the direction is from origin outwards)
    return spherical_to_cart(theta=np.deg2rad(random_zenith), phi=np.deg2rad(random_azimuth))


class IsotropicPhotonGenerator:
    """
    Creates random photon position together with its direction so that it ends up in the reactor front face.
    This use the custom MyLight as with the standard pvtrace.Light position and direction cannot be set together.
    """

    def __init__(self, tilt_angle):
        self.tilt_angle = tilt_angle
        self.base_position_generator = LightPosition(tilt_angle)

    def __call__(self, *args, **kwargs):
        position = self.base_position_generator()
        direction = create_diffuse_photon(self.tilt_angle)
        position += direction  # Translate position to ensure origin is not on reactor surface (+ visualization reasons)
        reversed_direction = tuple(
            -value for value in direction
        )  # Reversed to point towards the reactor!
        return position, reversed_direction
