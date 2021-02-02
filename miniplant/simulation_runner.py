"""
Module to set up a scene to run a simulation in direct or diffuse conditions
"""
import logging
import collections

from typing import Callable

from pvtrace import photon_tracer, MeshcatRenderer, Event, Scene

from miniplant.scene_creator import create_direct_scene, create_diffuse_scene

logger = logging.getLogger("pvtrace").getChild("miniplant")


def _common_simulation_runner(
    scene: Scene, num_photons: int = 100, render: bool = False, workers: int = None
):
    logger.debug(f"Starting ray-tracing with {num_photons} photons (Render is {render})")

    if render and workers > 1:
        raise RuntimeError("Sorry, cannot use renderer if more than 1 worker is used!")

    # SINGLE-THREADED
    if workers == 1:
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
    else:
        if render:
            logger.warning(
                "Cannot use renderer in multi-threaded simulations! Set workers to 1 for single-theaded"
            )
        # MULTI-THREADED
        results = scene.simulate(num_rays=num_photons, workers=workers)
        finals = [photon[-1][1] for photon in results]

    count_events = collections.Counter(finals)
    reacted_fraction = count_events[Event.REACT] / num_photons
    logger.debug(f"*** SIMULATION ENDED *** (Efficiency was {reacted_fraction:.3f})")
    return reacted_fraction


def run_direct_simulation(
    tilt_angle: int = 0,
    solar_elevation: int = 30,
    solar_azimuth: int = 180,
    solar_spectrum_function: Callable = lambda: 555,
    num_photons: int = 100,
    render: bool = False,
    workers: int = None,
) -> float:
    # Create scene with the provided parameters
    scene = create_direct_scene(
        tilt_angle=tilt_angle,
        solar_elevation=solar_elevation,
        solar_azimuth=solar_azimuth,
        solar_spectrum_function=solar_spectrum_function,
    )
    return _common_simulation_runner(scene, num_photons, render, workers)


def run_diffuse_simulation(
    tilt_angle: int = 0,
    solar_spectrum_function: Callable = lambda: 555,
    num_photons: int = 100,
    render: bool = False,
    workers: int = None,
) -> float:
    # Create scene with the provided parameters
    scene = create_diffuse_scene(
        tilt_angle=tilt_angle, solar_spectrum_function=solar_spectrum_function
    )
    return _common_simulation_runner(scene, num_photons, render, workers)


if __name__ == "__main__":
    run_diffuse_simulation(tilt_angle=60, render=True, workers=1, num_photons=400)
    # run_direct_simulation(tilt_angle=10, render=True, workers=1)
    input()
