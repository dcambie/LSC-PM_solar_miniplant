from pvtrace import *
import logging
import collections
from miniplant_reactor import create_standard_scene

# Set loggers
logging.getLogger('trimesh').disabled = True
logging.getLogger('shapely.geos').disabled = True
logging.getLogger("pvtrace").setLevel(logging.INFO)

LOGGER = logging.getLogger("pvtrace").getChild("miniplant")

# Local settings
TOTAL_PHOTONS = 100
RENDER = False


def run_simulation(location, time, num_photons: int = 100, render: bool = False) -> float:
    
    scene = create_standard_scene(tilt_angle=20, solar_elevation=40, solar_azimuth=180)

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
    LOGGER.info(f"Efficiency is {efficiency:.3f}")
    return efficiency
