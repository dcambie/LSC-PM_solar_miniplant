from pvtrace import *
import logging
import collections
from miniplant_reactor import create_standard_scene

# Set loggers
logging.getLogger('trimesh').disabled = True
logging.getLogger('shapely.geos').disabled = True
logging.getLogger("pvtrace").setLevel(logging.WARN)

LOGGER = logging.getLogger("pvtrace").getChild("miniplant")

# Local settings
TOTAL_PHOTONS = 100
RENDER = False


def run_simulation(tilt_angle: int = 0, solar_elevation: int = 30, solar_azimuth: int = 180,
                   num_photons: int = 100, render: bool = False) -> float:
    # Create scene with the provided parameters
    scene = create_standard_scene(tilt_angle=tilt_angle, solar_elevation=solar_elevation, solar_azimuth=solar_azimuth)

    if render:
        renderer = MeshcatRenderer(wireframe=False, open_browser=False)
        renderer.render(scene)
    finals = []

    for ray in scene.emit(num_photons):
        steps = photon_tracer.follow(scene, ray)
        path, events = zip(*steps)
        finals.append(events[-1])
        if render:
            renderer.add_ray_path(path)

    count_events = collections.Counter(finals)
    efficiency = count_events[Event.REACT] / num_photons
    LOGGER.info(f"Efficiency is {efficiency:.3f}")
    return efficiency
