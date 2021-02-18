import logging
from typing import Callable

import pandas as pd
import numpy as np

from pvtrace import (
    Node,
    Box,
    Sphere,
    Material,
    Luminophore,
    Absorber,
    Cylinder,
    Reactor,
    Light,
    Scene,
)
from pvtrace import isotropic
from pvtrace.material.utils import spherical_to_cart

# Experimental data
from miniplant.utils import (
    MyLight,
    VectorInverter,
    LightPosition,
    IsotropicPhotonGenerator,
)

REACTOR_AREA_IN_M2 = 0.47 * 0.47
MB_ABS_DATAFILE = "reactor_data/MB_1M_1m_ACN.txt"
LR305_ABS_DATAFILE = "reactor_data/Evonik_lr305_normalized_to_1m.txt"
LR305_EMS_DATAFILE = "reactor_data/Evonik_lr305_normalized_to_1m_ems.txt"

# Refractive indexes
PMMA_RI = 1.48
PFA_RI = 1.34
ACN_RI = 1.344

# Units
INCH = 0.0254  # meters


def _create_scene_common(tilt_angle, light_source, include_dye=None) -> Scene:
    logger = logging.getLogger("pvtrace").getChild("miniplant")
    logger.debug(f"Creating simulation scene w/ angle={tilt_angle}deg...")

    # Add nodes to the scene graph
    # Let's start with world - i.e. outer bounds
    world = Node(
        name="World (air)",
        geometry=Sphere(radius=10.0, material=Material(refractive_index=1.0)),
    )

    # Bind the light source to the current world
    light_source.parent = world

    # LSC-PM matrix
    matrix_component = [Absorber(coefficient=0.1)]  # PMMA background absorption

    if include_dye is None:
        include_dye = True

    if include_dye:
        matrix_component.append(Luminophore(
            coefficient=pd.read_csv(LR305_ABS_DATAFILE, sep="\t").values,
            emission=pd.read_csv(LR305_EMS_DATAFILE, sep="\t").values,
            quantum_yield=0.95,
            phase_function=isotropic,
        ))

    # LSC object
    reactor = Node(
        name="LSC-PM Reactor 47x47 cm^2",
        geometry=Box(
            size=(0.47, 0.47, 0.008),
            material=Material(
                refractive_index=PMMA_RI,
                components=matrix_component,
            ),
        ),
        parent=world,
    )

    # Now we need to populate the LSC with the capillaries, that are made by outer tubing and reaction mixture
    capillary = []
    r_mix = []

    # Reaction Mixture absorption
    reaction_absorption_coefficient = pd.read_csv(MB_ABS_DATAFILE, sep="\t").values
    reaction_mixture_material = Reactor(reaction_absorption_coefficient)

    # Create PFA 1/8" capillaries and their reaction mixture
    for capillary_num in range(16):
        capillary.append(
            Node(
                name=f"Capillary_PFA_{capillary_num}",
                geometry=Cylinder(
                    length=0.47,
                    radius=(1 / 8 * INCH) / 2,
                    material=Material(
                        refractive_index=PFA_RI,
                        components=[Absorber(coefficient=0.1)],  # PFA background absorption
                    ),
                ),
                parent=reactor,
            )
        )

        r_mix.append(
            Node(
                name=f"Reaction_mixture_{capillary_num}",
                geometry=Cylinder(
                    length=0.47,
                    radius=(1 / 16 * INCH) / 2,
                    material=Material(
                        refractive_index=ACN_RI, components=[reaction_mixture_material]
                    ),
                ),
                parent=capillary[-1],
            )
        )

        # Rotate capillary (w/ r_mix) so that is in LSC (default is Z axis)
        capillary[-1].rotate(np.radians(90), (1, 0, 0))
        # Adjust capillary position
        capillary[-1].translate((-0.47 / 2 + 0.01 + 0.03 * capillary_num, 0, 0))

    # Apply tilt angle to the reactor (and its children)
    reactor.rotate(np.radians(tilt_angle), (0, 1, 0))
    reactor.translate(
        (
            -np.sin(np.deg2rad(tilt_angle)) * 0.5 * 0.008,
            0,
            -np.cos(np.deg2rad(tilt_angle)) * 0.5 * 0.008,
        )
    )

    return Scene(world)


def create_direct_scene(
    tilt_angle: float = 30,
    solar_elevation: float = 30,
    solar_azimuth: float = 180,
    solar_spectrum_function: Callable = lambda: 555,
    include_dye: bool = None
) -> Scene:
    """ Create a scene with a fixed light position and direction, to match direct irradiation """

    # Define rays direction based on solar position
    solar_light_vector = spherical_to_cart(
        np.deg2rad(-solar_elevation + 90), np.deg2rad(-solar_azimuth + 180)
    )

    reversed_solar_light_vector = VectorInverter(solar_light_vector)

    # Create light
    solar_light = Node(
        name="Solar Light",
        light=Light(
            wavelength=solar_spectrum_function,
            direction=reversed_solar_light_vector,
            position=LightPosition(tilt_angle=tilt_angle),
        ),
        parent=None,
    )
    solar_light.translate(solar_light_vector)

    return _create_scene_common(tilt_angle=tilt_angle, light_source=solar_light, include_dye=include_dye)


def create_diffuse_scene(tilt_angle: float = 30, solar_spectrum_function: Callable = lambda: 555,
                         include_dye: bool = None):
    """ Create a scene with a random light position, to match diffuse irradiation """

    solar_light = Node(
        name="Solar Light",
        light=MyLight(
            wavelength=solar_spectrum_function,
            position_and_direction=IsotropicPhotonGenerator(tilt_angle),
        ),
        parent=None,
    )
    return _create_scene_common(tilt_angle=tilt_angle, light_source=solar_light, include_dye=include_dye)
