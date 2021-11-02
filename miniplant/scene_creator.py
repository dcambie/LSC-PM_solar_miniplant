import copy
import io
import logging
from typing import Callable

from pvtrace.material.utils import lambertian
from pvtrace.material.utils import cone

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
    Distribution,
)
from pvtrace import isotropic
from pvtrace.material.utils import spherical_to_cart

import pkgutil
from miniplant import reactor_data

MB_ABS_DATAFILE = pkgutil.get_data(__name__, "reactor_data/MB_1M_1m_ACN.tsv")
LR305_ABS_DATAFILE = pkgutil.get_data(__name__, "reactor_data/Evonik_lr305_normalized_to_1m.tsv")
LR305_EMS_DATAFILE = pkgutil.get_data(__name__, "reactor_data/Evonik_lr305_normalized_to_1m_ems.tsv")
WHITE_LED_EMS_DATAFILE = pkgutil.get_data(__name__, "reactor_data/White_LED_em.txt")

# Refractive indexes
BF33_RI = 1.47
PFA_RI = 1.34
ACN_RI = 1.344
Coating_RI = 1.50

# Units
INCH = 0.0254  # meter

# LTF reactor geometrical values
LTF_W = 60e-3
LTF_L = 101e-3
t_coating = 18e-6
LTF_H = 6e-3

# REACTOR_AREA_IN_M2 = LTF_W * LTF_L

wire_frame = False


def down_facing_LED():
    """
    Gets Lambertian direction and change Y axis
    """
    coord = lambertian()
    return coord[0], coord[1], -coord[2]


emission = pd.read_csv(io.BytesIO(WHITE_LED_EMS_DATAFILE), encoding="utf8", sep="\t").values
dist = Distribution(x=emission[:, 0], y=emission[:, 1])


def wavelength_LED():
    return dist.sample(np.random.uniform())


def _create_scene_common(light_source, include_dye=None) -> Scene:
    logger = logging.getLogger("pvtrace").getChild("miniplant")
    logger.debug(f"Creating simulation scene.")

    # Add nodes to the scene graph
    # Let's start with world - i.e. outer bounds
    world = Node(
        name="World (air)",
        geometry=Sphere(radius=1.0, material=Material(refractive_index=1.0)),
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
            ),
        ),
        appearance=dict(color=0xFFFFFF,
                        transparent=True,
                        opacity=1,
                        wireframe=True),
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
            ),
        ),
        appearance=coating_vis_prop,
        parent=reactor,
    )

    lower_coating = Node(
        name="Bottom coating",
        geometry=Box(
            size=(LTF_W, LTF_L, t_coating),
            material=Material(
                refractive_index=Coating_RI,
                components=coating_component,
            ),
        ),
        appearance=coating_vis_prop,
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
            trimesh=trimesh.load('reactor_data/LTF_Channels.stl'),
            material=Material(
                refractive_index=ACN_RI,
                components=[reaction_mixture_material]
            ),
        ),
        appearance=dict(color=0x0000FF,
                transparent=True,
                opacity=0.8,
                wireframe=True),
        parent=reactor,
    )

    ltf.translate((0, -2.32400e-3, 0))

    reactor.translate((0, 0, -1 / 2 * LTF_H - t_coating))

    return Scene(world)


def create_direct_scene(
        light_distribution: Callable = lambda: 555,
        include_dye: bool = None
) -> Scene:
    """ Create a scene with a fixed light position and direction, to match direct irradiation """

    m = 39
    n = 29
    dy = 1.3e-2
    dx = 1.7e-2
    h_box = 29.8e-2
    LED_POSITION = [(-np.ceil(n / 2) * dx + i * dx,
                     -np.ceil(m / 2) * dy + j * dy,
                     h_box) for i in range(n) for j in range(m)]

    def LED_pos():
        return random.choice(LED_POSITION)

    # Create light
    LED_light = Node(
        name="Solar Light",
        light=Light(
            wavelength=lambda: wavelength_LED(),
            direction=down_facing_LED,
            position=LED_pos,
        ),
        parent=None,
    )

    return _create_scene_common(light_source=LED_light, include_dye=include_dye)
