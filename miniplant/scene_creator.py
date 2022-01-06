import copy
import io
import logging
from typing import Callable

from pvtrace.material.utils import cone, lambertian

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

import pkgutil
from miniplant import reactor_data

MB_ABS_DATAFILE = pkgutil.get_data(__name__, "reactor_data/MB_1M_1m_ACN.tsv")
LR305_ABS_DATAFILE = pkgutil.get_data(__name__, "reactor_data/Red_Absorption.txt")
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

# LED emission data
emission = pd.read_csv(io.BytesIO(WHITE_LED_EMS_DATAFILE), encoding="utf8", sep="\t").values
dist = Distribution(x=emission[:, 0], y=emission[:, 1])

# LED positions for the LED Box, in total 1170 LEDs.
m = 39
n = 30
dx = 1.3e-2
dy = 1.7e-2
h_box = 29.8e-2
LED_POSITION = [(-m / 2 * dx + (i + 0.5) * dx,
                 -n / 2 * dy + (j + 0.5) * dy,
                 h_box) for i in range(m) for j in range(n)]


def wavelength_led():
    return dist.sample(np.random.uniform())


def led_pos():
    return random.choice(LED_POSITION)


def down_facing_led():
    """
    Gets Lambertian or conical direction and change Y axis
    """
    # coord = lambertian()
    coord = cone(np.deg2rad(66))
    return coord[0], coord[1], -coord[2]


def create_led_scene() -> Scene:
    world = Node(
        name="World (air)",
        geometry=Sphere(radius=1.0),
    )

    leds = []

    # Create LEDs for visualisation
    for led_num in range(len(LED_POSITION)):
        leds.append(
            Node(
                name=f"LED_render_{led_num}",
                geometry=Cylinder(
                    length=0,
                    radius=2.5e-3,
                    material=None
                ),
                appearance=dict(color=0x000000,
                                ),
                parent=world,
            )
        )
        # # Adjust capillary position
        leds[-1].translate((LED_POSITION[led_num][0], LED_POSITION[led_num][1], LED_POSITION[led_num][2]))

    return Scene(world)


def _create_scene_common(light_source, include_coating=None, include_dye=None) -> Scene:
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

    # Glass matrix and appearance
    glass_component = [Absorber(coefficient=0.1)]  # PMMA background absorption
    glass_color = 0xFFFFFF
    glass_vis_prop = dict(transparent=False,
                          opacity=0.8,
                          wireframe=True,
                          )
    channel_color = 0x0000FF
    channel_vis_prop = dict(transparent=True,
                            opacity=0.9,
                            wireframe=True)

    # Glass reactor object
    reactor = Node(
        name="LTF Reactor",
        geometry=Box(
            size=(LTF_W, LTF_L, LTF_H),
            material=Material(
                refractive_index=BF33_RI,
                components=glass_component,
            ),
        ),
        appearance=dict(color=glass_color,
                        ),
        parent=world,
    )
    reactor.appearance["meshcat"] = glass_vis_prop

    # Reaction Mixture absorption
    reaction_absorption_coefficient = pd.read_csv(io.BytesIO(MB_ABS_DATAFILE), encoding="utf8", sep="\t").values
    reaction_mixture_material = Reactor(reaction_absorption_coefficient)

    # Inserting reaction mixture channels in the glass reactor object
    ltf = Node(
        name='LTF channels',
        geometry=Mesh(
            trimesh=trimesh.load('reactor_data/LTF_Channels.stl'),
            material=Material(
                refractive_index=ACN_RI,
                components=[reaction_mixture_material]
            ),
        ),
        appearance=dict(color=channel_color,
                        ),
        parent=reactor,
    )
    ltf.appearance["meshcat"] = channel_vis_prop
    ltf.translate((0, -2.32400e-3, 0))

    if include_coating is None:
        include_coating = True

    if include_dye is None:
        include_dye = True

    if include_dye is True and include_coating is False:
        raise RuntimeError("Sorry, cannot enable dye without a coating!")

    if include_coating:
        t_coating_factor = 1
        # Coating matrix and appearance
        coating_component = [Absorber(coefficient=0.1)]  # PMMA background absorption
        coating_color = 0xFFFFFF
        coating_vis_prop = dict(transparent=True,
                                opacity=0.3,
                                wireframe=False,
                                )
        if include_dye:
            coating_color = 0xFF0000
            coating_component.append(Luminophore(
                coefficient=pd.read_csv(io.BytesIO(LR305_ABS_DATAFILE), encoding="utf8", sep="\t").values,
                emission=pd.read_csv(io.BytesIO(LR305_EMS_DATAFILE), encoding="utf8", sep="\t").values,
                quantum_yield=0.95,
                phase_function=isotropic,
            ))

        upper_coating = Node(
            name="Top coating",
            geometry=Box(
                size=(LTF_W, LTF_L, t_coating),
                material=Material(
                    refractive_index=Coating_RI,
                    components=coating_component,
                ),
            ),
            appearance=dict(color=coating_color,
                            ),
            parent=reactor,
        )
        upper_coating.appearance["meshcat"] = coating_vis_prop

        lower_coating = Node(
            name="Bottom coating",
            geometry=Box(
                size=(LTF_W, LTF_L, t_coating),
                material=Material(
                    refractive_index=Coating_RI,
                    components=coating_component,
                ),
            ),
            appearance=dict(color=coating_color,
                            ),
            parent=reactor,
        )
        lower_coating.appearance["meshcat"] = coating_vis_prop

        upper_coating.translate((0, 0, 1 / 2 * (LTF_H + t_coating)))
        lower_coating.translate((0, 0, -1 / 2 * (LTF_H + t_coating)))
    else:
        t_coating_factor = 0

    reactor.translate((0, 0, -1 / 2 * LTF_H - t_coating_factor*t_coating))

    return Scene(world)


def create_direct_scene(
        include_coating: bool = True,
        include_dye: bool = None
) -> Scene:
    """ Create a scene with fixed light positions and directions, to match LED Box irradiation """
    # Create light
    led_light = Node(
        name="Solar Light",
        light=Light(
            wavelength=lambda: wavelength_led(),
            direction=down_facing_led,
            position=led_pos,
        ),
        parent=None,
    )

    return _create_scene_common(light_source=led_light, include_coating=include_coating, include_dye=include_dye)
