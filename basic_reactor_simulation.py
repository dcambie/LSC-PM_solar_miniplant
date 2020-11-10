from pvtrace import *
import numpy as np
import pandas as pd
import logging
import time
import collections

from pvtrace.geometry.transformations import rotation_matrix
from pvtrace.material.component import Reactor
from pvtrace.material.utils import spherical_to_cart

# Set loggers
logging.getLogger('trimesh').disabled = True
logging.getLogger('shapely.geos').disabled = True
logging.getLogger("pvtrace").setLevel(logging.INFO)

# Experimental data
MB_ABS_DATAFILE = "reactor_data/MB_1M_1m_ACN.txt"
LR305_ABS_DATAFILE = "reactor_data/Evonik_lr305_normalized_to_1m.txt"
LR305_EMS_DATAFILE = "reactor_data/Evonik_lr305_normalized_to_1m_ems.txt"

# Refractive indexes
PMMA_RI = 1.48
PFA_RI = 1.34
ACN_RI = 1.344
INCH = 0.0254  # meters

# LSCPM tilt angle
TILT_ANGLE = 30

# Sun position
AZIMUT = 180
ELEVATION = 38

# Local settings
TOTAL_PHOTONS = 100
RENDER = False

# Add nodes to the scene graph
world = Node(
    name="World (air)",
    geometry=Sphere(radius=10.0, material=Material(refractive_index=1.0))
)

# LSC object
reactor = Node(
    name="LSC-PM Reactor 47x47 cm^2",
    geometry=Box(
        size=(0.47, 0.47, 0.008),
        material=Material(
            refractive_index=PMMA_RI,
            components=[
                Luminophore(
                    coefficient=pd.read_csv(LR305_ABS_DATAFILE, sep="\t").values,
                    emission=pd.read_csv(LR305_EMS_DATAFILE, sep="\t").values,
                    quantum_yield=0.95,
                    phase_function=isotropic
                ),
                Absorber(coefficient=0.1)  # PMMA background absorption
            ]
        ),
    ),
    parent=world,
)

capillary = []
r_mix = []
epsilon = 1e-6

# Reaction Mixture absorption
reaction_absorption_coefficient = pd.read_csv(MB_ABS_DATAFILE, sep="\t").values
reaction_mixture_material = Reactor(reaction_absorption_coefficient)

for capillary_num in range(16):
    capillary.append(
        Node(
            name=f"Capillary_PFA_{capillary_num}",
            geometry=Cylinder(
                length=0.47,
                radius=(1/8*INCH)/2,
                material=Material(
                    refractive_index=PFA_RI,
                    components=[
                        Absorber(coefficient=0.1)  # PFA background absorption
                    ]
                ),
            ),
            parent=reactor,
            # opacity=0.3
        )
    )

    r_mix.append(
        Node(
            name=f"Reaction_mixture_{capillary_num}",
            geometry=Cylinder(
                length=0.47,
                radius=(1/16*INCH)/2,
                material=Material(
                    refractive_index=ACN_RI,
                    components=[reaction_mixture_material]
                ),
            ),
            parent=capillary[-1],
            # opacity=1,
            # color=0X0000FF
        )
    )

    # Rotate capillary (w/ r_mix) so that is in LSC (default is Z axis)
    capillary[-1].rotate(np.radians(90), (1, 0, 0))
    # Adjust capillary position
    capillary[-1].translate((-0.47/2+0.01+0.03*capillary_num, 0, 0))

# Azimut and elevation are converted to the cartesian reference system used in simulations.
vector = spherical_to_cart(np.radians(-ELEVATION+90), np.radians(-AZIMUT+180))


def reversed_direction():
    return tuple(-value for value in vector)


def light_position():
    position = rectangular_mask(0.47/2, 0.47/2)
    matrix = np.linalg.inv(rotation_matrix(np.radians(-30), (0, 1, 0)))

    homogeneous_pt = np.ones(4)
    homogeneous_pt[0:3] = position
    new_pt = np.dot(matrix, homogeneous_pt)[0:3]
    return tuple(new_pt)


def solar():
    return 555


solar_light = Node(
    name="Solar Light",
    light=Light(
        wavelength=solar,
        direction=reversed_direction,
        position=light_position
    ),
    parent=world
)
solar_light.translate(vector)

# Apply tilt angle
reactor.rotate(np.radians(TILT_ANGLE), (0, 1, 0))

scene = Scene(world)
if RENDER:
    renderer = MeshcatRenderer(wireframe=False, open_browser=False)
    renderer.render(scene)
finals = []
count = 0

for ray in scene.emit(TOTAL_PHOTONS):
    count += 1
    steps = photon_tracer.follow(scene, ray)
    path, events = zip(*steps)
    finals.append(events[-1])
    if RENDER:
        renderer.add_ray_path(path)
    if count % 100 == 0:
        print(count)

count_events = collections.Counter(finals)
efficiency = count_events[Event.REACT] / TOTAL_PHOTONS
print(f"Efficiency is {efficiency:.3f}")


# Wait for Ctrl-C to terminate the script; keep the window open
print("Press Enter to close")
input()
