import io
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

import pkgutil

REACTOR_AREA_IN_M2 = 0.47 * 0.47

MB_ABS_DATAFILE = pkgutil.get_data(__name__, "reactor_data/MB_1M_1m_ACN.tsv")
LR305_ABS_DATAFILE = pkgutil.get_data(
    __name__, "reactor_data/Evonik_lr305_normalized_to_1m.tsv"
)
LR305_EMS_DATAFILE = pkgutil.get_data(
    __name__, "reactor_data/Evonik_lr305_normalized_to_1m_ems.tsv"
)

# Refractive indexes
PMMA_RI = 1.48
PFA_RI = 1.34
ACN_RI = 1.344

# Units
INCH = 0.0254  # meters


def _create_scene_common(tilt_angle, light_source, include_dye=None, **kwargs) -> Scene:
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
        matrix_component.append(
            Luminophore(
                coefficient=pd.read_csv(
                    io.BytesIO(LR305_ABS_DATAFILE), encoding="utf8", sep="\t"
                ).values,
                emission=pd.read_csv(
                    io.BytesIO(LR305_EMS_DATAFILE), encoding="utf8", sep="\t"
                ).values,
                quantum_yield=0.95,
                phase_function=isotropic,
            )
        )

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

    if kwargs.get("add_bottom_PV", False):
        bottomPV = Node(
            name="bottomPV",
            geometry=Box(
                size=(0.47, 0.47, 0.008),
                material=Material(
                    refractive_index=3.4,
                    components=[Absorber(coefficient=1e10)],
                    color=0x000000,
                    transparent=False,
                    opacity=0.5,
                ),
            ),
            parent=reactor,
        )
        bottomPV.translate(
            (
                0,
                0,
                -0.025,
            )
        )

    if kwargs.get("add_side_PV", False):
        side1 = Node(
            name="sidePV1",
            geometry=Box(
                size=(0.47, 0.01, 0.008),
                material=Material(
                    refractive_index=3.4,
                    components=[Absorber(coefficient=1e10)],
                    color=0x000000,
                    transparent=False,
                    opacity=0.5,
                ),
            ),
            parent=reactor,
        )
        side1.translate(
            (
                0,
                0.47 / 2 + 0.005,
                0,
            )
        )
        side2 = Node(
            name="sidePV2",
            geometry=Box(
                size=(0.47, 0.01, 0.008),
                material=Material(
                    refractive_index=3.4,
                    components=[Absorber(coefficient=1e10)],
                    color=0x000000,
                    transparent=False,
                    opacity=0.5,
                ),
            ),
            parent=reactor,
        )
        side2.translate(
            (
                0,
                -0.47 / 2 - 0.005,
                0,
            )
        )
        side3 = Node(
            name="sidePV3",
            geometry=Box(
                size=(0.01, 0.47, 0.008),
                material=Material(
                    refractive_index=3.4,
                    components=[Absorber(coefficient=1e10)],
                    color=0x000000,
                    transparent=False,
                    opacity=0.5,
                ),
            ),
            parent=reactor,
        )
        side3.translate(
            (
                0.47 / 2 + 0.005,
                0,
                0,
            )
        )
        side4 = Node(
            name="sidePV4",
            geometry=Box(
                size=(0.01, 0.47, 0.008),
                material=Material(
                    refractive_index=3.4,
                    components=[Absorber(coefficient=1e10)],
                    color=0x000000,
                    transparent=False,
                    opacity=0.5,
                ),
            ),
            parent=reactor,
        )
        side4.translate(
            (
                -0.47 / 2 - 0.005,
                0,
                0,
            )
        )

    # Now we need to populate the LSC with the capillaries, that are made by outer tubing and reaction mixture
    capillary = []
    r_mix = []

    # Reaction Mixture absorption
    reaction_absorption_coefficient = pd.read_csv(
        io.BytesIO(MB_ABS_DATAFILE), encoding="utf8", sep="\t"
    ).values
    reaction_mixture_material = Reactor(reaction_absorption_coefficient)

    # Create PFA 1/8" capillaries and their reaction mixture
    pfa_cil = Cylinder(
        length=0.47,
        radius=(1 / 8 * INCH) / 2,
        material=Material(
            refractive_index=PFA_RI,
            components=[Absorber(coefficient=0.1)],  # PFA background absorption
        ),
    )
    pfa_cil.color = 0xEEEEEE
    pfa_cil.transparency = True
    pfa_cil.opacity = 0.5

    for capillary_num in range(16):
        capillary.append(
            Node(
                name=f"Capillary_PFA_{capillary_num}",
                geometry=pfa_cil,
                parent=reactor,
            )
        )

        reaction_cil = Cylinder(
            length=0.47,
            radius=(1 / 16 * INCH) / 2,
            material=Material(
                refractive_index=ACN_RI, components=[reaction_mixture_material]
            ),
        )
        reaction_cil.color = 0x0000FF
        reaction_cil.transparency = False
        reaction_cil.opacity = 1

        r_mix.append(
            Node(
                name=f"Reaction_mixture_{capillary_num}",
                geometry=reaction_cil,
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
    include_dye: bool = None,
    **kwargs,
) -> Scene:
    """Create a scene with a fixed light position and direction, to match direct irradiation"""

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

    return _create_scene_common(
        tilt_angle=tilt_angle,
        light_source=solar_light,
        include_dye=include_dye,
        **kwargs,
    )


def create_diffuse_scene(
    tilt_angle: float = 30,
    solar_spectrum_function: Callable = lambda: 555,
    include_dye: bool = None,
):
    """Create a scene with a random light position, to match diffuse irradiation"""

    solar_light = Node(
        name="Solar Light",
        light=MyLight(
            wavelength=solar_spectrum_function,
            position_and_direction=IsotropicPhotonGenerator(tilt_angle),
        ),
        parent=None,
    )
    return _create_scene_common(
        tilt_angle=tilt_angle, light_source=solar_light, include_dye=include_dye
    )
