import io
import math

import pandas as pd
import numpy as np
from meshcat import Visualizer

from pvtrace import Node, Sphere, Material, Absorber, Luminophore, isotropic, Reactor, Cylinder, Box, Scene, \
    MeshcatRenderer

import meshcat.transformations as tf
from meshcat.animation import Animation, AnimationFrameVisualizer
import meshcat

from miniplant.scene_creator import LR305_ABS_DATAFILE, LR305_EMS_DATAFILE, PMMA_RI, MB_ABS_DATAFILE, INCH, PFA_RI, \
    MeOH_RI
from miniplant.utils import MyLight, IsotropicPhotonGenerator


TILT_ANGLE = 30

world = Node(
    name="Air",
    geometry=Sphere(radius=10.0, material=Material(refractive_index=1.0)),
)


def green_photons():
    return 555


light_source = Node(
    name="Solar Light",
    light=MyLight(
        wavelength=green_photons,
        position_and_direction=IsotropicPhotonGenerator(TILT_ANGLE),
    ),
    parent=world
)

# LSC-PM matrix
matrix_component = [
    Absorber(coefficient=0.1),  # PMMA background absorption
    Luminophore(
        coefficient=pd.read_csv(io.BytesIO(LR305_ABS_DATAFILE), encoding="utf8", sep="\t").values,
        emission=pd.read_csv(io.BytesIO(LR305_EMS_DATAFILE), encoding="utf8", sep="\t").values,
        quantum_yield=0.95,
        phase_function=isotropic,
    )
]

# LSC object
reactor = Node(
    name="LSC-PM",
    geometry=Box(
        size=(0.47, 0.47, 0.008),
        material=Material(
            refractive_index=PMMA_RI,
            components=matrix_component,
            color=0xff0000,
            transparent=True,
            opacity=0.6,
            reflectivity=0
        ),
    ),
    parent=world,
)

# Now we need to populate the LSC with the capillaries, that are made by outer tubing and reaction mixture
capillary = []
r_mix = []

# Reaction Mixture absorption
reaction_absorption_coefficient = pd.read_csv(io.BytesIO(MB_ABS_DATAFILE), encoding="utf8", sep="\t").values
reaction_mixture_material = Reactor(reaction_absorption_coefficient)

# Create PFA 1/8" capillaries and their reaction mixture
for capillary_num in range(16):
    capillary.append(
        Node(
            name=f"Capillary_{capillary_num}",
            geometry=Cylinder(
                length=0.47,
                radius=(1 / 8 * INCH) / 2,
                material=Material(
                    refractive_index=PFA_RI,
                    components=[Absorber(coefficient=0.1)],  # PFA background absorption
                    color = 0xeeeeee,
                    transparent = True,
                    opacity = 0.5,
                    reflectivity=0.1
                    ),
                ),
            parent=reactor,
        )
    )

    r_mix.append(
        Node(
            name=f"Rx_{capillary_num}",
            geometry=Cylinder(
                length=0.47,
                radius=(1 / 16 * INCH) / 2,
                material=Material(
                    refractive_index=MeOH_RI,
                    components=[reaction_mixture_material],
                    color = 0x0000ff,
                    transparent = False,
                    opacity = 1
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
reactor.rotate(np.radians(TILT_ANGLE), (0, 1, 0))
reactor.translate(
    (
        -np.sin(np.deg2rad(TILT_ANGLE)) * 0.5 * 0.008,
        0,
        -np.cos(np.deg2rad(TILT_ANGLE)) * 0.5 * 0.008,
    )
)

floor = Node(
    name = "floor",
    geometry=Box(
        size=(20, 20, 0.001),
        material=Material(refractive_index=10, components=[Absorber(coefficient=1e10)],
            color=0x000000,
            transparent = True,
            opacity = 0.5,
            ),
    ),
    parent = world
)
half_reactor_vertical_projection = (1/2) * 0.47 * np.sin(np.deg2rad(TILT_ANGLE))
thickness = 0.008 * np.cos(np.deg2rad(TILT_ANGLE))
floor.translate((0, 0, -half_reactor_vertical_projection -thickness))

scene = Scene(world)


renderer = MeshcatRenderer(open_browser=True)
renderer.render(scene)

# Hide axis
axes = renderer.vis["/Axes"].set_property("visible", False)
# Show grid
grid = renderer.vis["/Grid"].set_property("visible", True)


# bg = renderer.vis["/Background"]
# bg.set_property("visible", False)

anim = Animation()

# bg = Visualizer.view_into(renderer.vis.window, meshcat.path.Path("/Background"))
# print(bg)
# bg.delete()
# renderer.vis["/Background"].delete()
# renderer.vis["/Cameras/default"].set_transform(tf.translation_matrix([0, 0, 1]))


# reactor.transformation_to(world)

camera_path = "/Cameras/default/rotated/<object>"


with anim.at_frame(renderer.vis, 0) as frame:
    frame[camera_path].set_property("zoom", "number", 50)
with anim.at_frame(renderer.vis, 60) as frame:
    blabla =frame[camera_path]
    frame[camera_path].set_property("zoom", "number", 1)

x = AnimationFrameVisualizer(anim, meshcat.path.Path(tuple("Cameras/default".split('/'))), 30)
x.set_transform(tf.translation_matrix([0, 1, 1]))
y = AnimationFrameVisualizer(anim, meshcat.path.Path(tuple("Cameras/default".split('/'))), 60)
y.set_transform(tf.translation_matrix([0, 0, 1]))


# with anim.at_frame(renderer.vis, 120) as frame:
#     frame["/Cameras/default"].set_transform(tf.translation_matrix([0, 1, 1]))
with anim.at_frame(renderer.vis, 120) as frame:
    frame[camera_path].set_property("zoom", "number", 2)

# While we're animating the camera zoom, we can also animate any other
# properties we want. Let's simultaneously translate the box during
# the same animation:
# with anim.at_frame(renderer.vis, 0) as frame:
#     frame["SUN"].set_transform(tf.translation_matrix([0, -1, 0]))

renderer.vis.set_animation(anim)
input()
