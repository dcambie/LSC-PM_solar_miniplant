from typing import Callable

from pvtrace import *
import logging
import collections
from miniplant_reactor import create_standard_scene

# Set loggers
logging.getLogger('trimesh').disabled = True
logging.getLogger('shapely.geos').disabled = True
logging.getLogger("pvtrace").setLevel(logging.DEBUG)

logger = logging.getLogger("pvtrace").getChild("miniplant")

# Local settings
TOTAL_PHOTONS = 100
RENDER = False


def run_simulation(tilt_angle: int = 0, solar_elevation: int = 30, solar_azimuth: int = 180,
                   solar_spectrum_function: Callable = lambda: 555,
                   num_photons: int = 100, render: bool = False) -> float:

    # Create scene with the provided parameters
    scene = create_standard_scene(tilt_angle=tilt_angle, solar_elevation=solar_elevation, solar_azimuth=solar_azimuth,
                                  solar_spectrum_function=solar_spectrum_function)

    if render:
        renderer = MeshcatRenderer(wireframe=False, open_browser=False)
        renderer.render(scene)
    finals = []

    logger.info(f"Starting ray-tracing with {num_photons} photons (Render is {render})")
    # for ray in scene.emit(num_photons):
    #     steps = photon_tracer.follow(scene, ray)
    #     path, events = zip(*steps)
    #     finals.append(events[-1])
    #     if render:
    #         renderer.add_ray_path(path)

    results = scene.simulate(num_rays=num_photons)
    all_workers_results = [item for sublist in results for item in sublist]
    finals = [photon[-1][1] for photon in all_workers_results]

    count_events = collections.Counter(finals)
    efficiency = count_events[Event.REACT] / num_photons
    logger.info(f"*** SIMULATION ENDED *** (Efficiency was {efficiency:.3f})")
    return efficiency
