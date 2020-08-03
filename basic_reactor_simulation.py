from pvtrace import *
import numpy as np
import pandas as pd
import logging
import time
import sys

from pvtrace.geometry.transformations import rotation_matrix
from pvtrace.material.utils import spherical_to_cart

# Logging
logging.getLogger('trimesh').disabled = True
logging.getLogger('shapely.geos').disabled = True
logging.getLogger("pvtrace").setLevel(logging.INFO)

# Experimental data
MB_ABS_DATAFILE = "reactor_data/MB_1M_1m_ACN.txt"
LR305_ABS_DATAFILE = "reactor_data/Evonik_lr305_normalized_to_1m.txt"
LR305_EMS_DATAFILE = "reactor_data/Evonik_lr305_normalized_to_1m_ems.txt"
PMMA_RI = 1.48
PFA_RI = 1.34
ACN_RI = 1.344
INCH = 0.0254

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
    color=0xFF0000,
    opacity=0.2
)
reactor.rotate(np.radians(30), (0, 1, 0))

capillary = []
r_mix = []
epsilon = 1e-6
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
            opacity=0.3
        )
    )

    # Reaction Mixture absorption
    reaction_absorption_coefficient = pd.read_csv(MB_ABS_DATAFILE, sep="\t").values
    x = reaction_absorption_coefficient[0]
    ri = np.ones(len(x)) * ACN_RI
    ACN_refractive_index = np.column_stack((x, ri))
    reaction_mixture_material = Absorber(ACN_refractive_index, reaction_absorption_coefficient)

    r_mix.append(
        Node(
            name=f"Reaction_mixture_{capillary_num}",
            geometry=Cylinder(
                length=0.47,
                radius=(1/16*INCH)/2,
                material=Material(
                    refractive_index=PFA_RI,
                    components=[reaction_mixture_material]
                ),
            ),
            parent=capillary[-1],
            opacity=1,
            color=0X0000FF
        )
    )

    capillary[-1].rotate(np.radians(90), (1, 0, 0))  # Rotate capillary (w/ r_mix) so that is in LSC (default is Z axis)
    capillary[-1].translate((-0.47/2+0.01+0.03*capillary_num, 0, 0))


vector = spherical_to_cart(np.radians(45), np.radians(60))


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

light = Node(
    name="Solar Light",
    light=Light(
        wavelength=solar,
        direction=reversed_direction,
        position=light_position
    ),
    parent=world
)
light.translate(vector)

renderer = MeshcatRenderer(wireframe=False, open_browser=True)
scene = Scene(world)
renderer.render(scene)
finals = []
for ray in scene.emit(100):
    steps = photon_tracer.follow(scene, ray)
    path, events = zip(*steps)
    finals.append(events[-1])
    renderer.add_ray_path(path)

absorbed = [event for event in finals if event == Event.ABSORB]
print(f"absorbed {len(absorbed)}")

# Wait for Ctrl-C to terminate the script; keep the window open
print("Ctrl-C to close")
while True:
    try:
        time.sleep(.3)
    except KeyboardInterrupt:
        sys.exit()
