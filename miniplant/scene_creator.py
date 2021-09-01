import copy
import io
import logging
from typing import Callable

from pvtrace.material.utils import lambertian

import pandas as pd
import numpy as np
import trimesh
import random

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
    Mesh,
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
from miniplant import reactor_data

MB_ABS_DATAFILE = pkgutil.get_data(__name__, "reactor_data/MB_1M_1m_ACN.tsv")
LR305_ABS_DATAFILE = pkgutil.get_data(__name__, "reactor_data/Evonik_lr305_normalized_to_1m.tsv")
LR305_EMS_DATAFILE = pkgutil.get_data(__name__, "reactor_data/Evonik_lr305_normalized_to_1m_ems.tsv")

# Refractive indexes
BF33_RI = 1.47
PFA_RI = 1.34
ACN_RI = 1.344
Coating_RI = 1.50

# To micro meters as a check for meshes
m_to_mu = 1.0

# Units
INCH = 0.0254  # meter

# LTF reactor geometrical values
LTF_W = 60e-3
LTF_L = 101e-3
t_coating = 2e-5
LTF_H = 6e-3

# REACTOR_AREA_IN_M2 = LTF_W * LTF_L

wire_frame = False


def downfacing_labertian():
    """
    Gets Lambertian direction and change Y axis
    """
    coord = lambertian()
    return coord[0], coord[1], -coord[2]


def _create_scene_common(tilt_angle, light_source, include_dye=None) -> Scene:
    logger = logging.getLogger("pvtrace").getChild("miniplant")
    logger.debug(f"Creating simulation scene w/ angle={tilt_angle}deg...")

    # Add nodes to the scene graph
    # Let's start with world - i.e. outer bounds
    world = Node(
        name="World (air)",
        geometry=Sphere(radius=10.0 * m_to_mu, material=Material(refractive_index=1.0)),
    )

    # Bind the light source to the current world
    light_source.parent = world

    # Glass matrix
    glass_component = [Absorber(coefficient=0.1)]  # PMMA background absorption

    # Coating matrix
    coating_component = [Absorber(coefficient=0.1)]  # PMMA background absorption

    if include_dye is None:
        include_dye = True

    if include_dye:
        coating_component.append(Luminophore(
            coefficient=pd.read_csv(io.BytesIO(LR305_ABS_DATAFILE), encoding="utf8", sep="\t").values,
            emission=pd.read_csv(io.BytesIO(LR305_EMS_DATAFILE), encoding="utf8", sep="\t").values,
            quantum_yield=0.95,
            phase_function=isotropic,
        ))

    # LSC object
    reactor = Node(
        name="LTF Reactor",
        geometry=Box(
            size=(LTF_W, LTF_L, LTF_H),
            material=Material(
                refractive_index=BF33_RI,
                components=glass_component,
                color=0xFFFFFF,
                transparent=True,
                opacity=1,
                wireframe=True,
            ),
        ),
        parent=world,
    )

    coating_vis_prop = dict(color=0xFF0000,
                            transparent=True,
                            opacity=0.5,
                            wireframe=wire_frame, )

    upper_coating = Node(
        name="Top coating",
        geometry=Box(
            size=(LTF_W, LTF_L, t_coating),
            material=Material(
                refractive_index=Coating_RI,
                components=coating_component,
                **coating_vis_prop,
            ),
        ),
        parent=reactor,
    )

    lower_coating = Node(
        name="Bottom coating",
        geometry=Box(
            size=(LTF_W, LTF_L, t_coating),
            material=Material(
                refractive_index=Coating_RI,
                components=coating_component,
                **coating_vis_prop,
            ),
        ),
        parent=reactor,
    )

    upper_coating.translate((0, 0, 1 / 2 * (LTF_H + t_coating)))
    lower_coating.translate((0, 0, -1 / 2 * (LTF_H + t_coating)))

    # Reaction Mixture absorption
    reaction_absorption_coefficient = pd.read_csv(io.BytesIO(MB_ABS_DATAFILE), encoding="utf8", sep="\t").values
    reaction_mixture_material = Reactor(reaction_absorption_coefficient)

    ltf = Node(
        name='LTF channels',
        geometry=Mesh(
            trimesh=trimesh.load('reactor_data/Cyl_Channels.stl'),
            material=Material(
                refractive_index=ACN_RI,
                components=[reaction_mixture_material],
                color=0x0000FF,
                transparent=True,
                opacity=0.8,
                wireframe=True,
            ),
        ),
        parent=reactor,
    )

    reactor.translate((0, 0, -1 / 2 * LTF_H - t_coating))

    # # Apply tilt angle to the reactor (and its children)
    # reactor.rotate(np.radians(tilt_angle), (0, 1, 0))
    # reactor.translate(
    #     (
    #         -np.sin(np.deg2rad(tilt_angle)) * 0.5 * 0.008,
    #         0,
    #         -np.cos(np.deg2rad(tilt_angle)) * 0.5 * 0.008,
    #     )
    # )

    return Scene(world)


def create_direct_scene(
        tilt_angle: float = 0,
        solar_elevation: float = 30,
        solar_azimuth: float = 180,
        solar_spectrum_function: Callable = lambda: 555,
        include_dye: bool = None
) -> Scene:
    """ Create a scene with a fixed light position and direction, to match direct irradiation """


    def led_pos():
        leds = [(0.01,0,0.1), (-0.01,0,0.1)]
        led = random.choice(leds)
        return led

    # Create light
    solar_light = Node(
        name="Solar Light",
        light=Light(
            wavelength=solar_spectrum_function,
            direction=downfacing_labertian,
            position=led_pos,
        ),
        parent=None,
    )


    return _create_scene_common(tilt_angle=tilt_angle, light_source=solar_light, include_dye=include_dye)


def create_diffuse_scene(tilt_angle: float = 0, solar_spectrum_function: Callable = lambda: 555,
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
