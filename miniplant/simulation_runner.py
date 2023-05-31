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
    scene: Scene,
    num_photons: int = 100,
    render: bool = False,
    workers: int = 1,
):
    logger.debug(
        f"Starting ray-tracing with {num_photons} photons (Render is {render})"
    )

    if render and workers > 1:
        raise RuntimeError("Sorry, cannot use renderer if more than 1 worker is used!")

    bottomPV_count = 0
    sidePV_count = 0
    photon_path = []
    # SINGLE-THREADED
    if workers == 1:
        if render:
            renderer = MeshcatRenderer(open_browser=True)
            renderer.render(scene)
        finals = []
        for ray in scene.emit(num_photons):
            steps = photon_tracer.follow(scene, ray)
            path, events = zip(*steps)
            finals.append(events[-1])
            if render:
                renderer.add_ray_path(path)

            from pvtrace.algorithm.photon_tracer import next_hit

            myray = steps[-1][0]
            intersect = next_hit(scene, myray)
            side_PV = {"sidePV1", "sidePV2", "sidePV3", "sidePV4"}
            if intersect and intersect[0].name == "bottomPV":
                bottomPV_count += 1
            if intersect and intersect[0].name in side_PV:
                sidePV_count += 1

            photon_path.append(path)
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
    if bottomPV_count > 0:
        logger.info(
            f"Bottom PV absorbed {bottomPV_count} photons (i.e. {bottomPV_count/num_photons * 100 :.2f} %) "
        )
    if sidePV_count > 0:
        logger.info(
            f"The side PV absorbed {sidePV_count} photons (i.e. {sidePV_count / num_photons * 100 :.2f} %) "
        )

    # This polymorphism in return type is bad, but the PV thing is only needed once and returning a list breaks things
    if bottomPV_count > 0 or sidePV_count > 0:
        return reacted_fraction, scene, photon_path, bottomPV_count, sidePV_count
    return reacted_fraction


def run_direct_simulation(
    tilt_angle: int = 0,
    solar_elevation: int = 30,
    solar_azimuth: int = 180,
    solar_spectrum_function: Callable = lambda: 555,
    num_photons: int = 100,
    render: bool = False,
    workers: int = 1,
    include_dye: bool = None,
    **kwargs,
):
    """
    Create a scene for direct irradiation with the provided parameters and runs a simulation on it
    """
    scene = create_direct_scene(
        tilt_angle=tilt_angle,
        solar_elevation=solar_elevation,
        solar_azimuth=solar_azimuth,
        solar_spectrum_function=solar_spectrum_function,
        include_dye=include_dye,
        **kwargs,
    )
    return _common_simulation_runner(scene, num_photons, render, workers)


def run_diffuse_simulation(
    tilt_angle: int = 0,
    solar_spectrum_function: Callable = lambda: 555,
    num_photons: int = 100,
    render: bool = False,
    workers: int = 1,
    include_dye: bool = None,
):
    """
    Create a scene for diffuse irradiation with the provided parameters and runs a simulation on it
    """
    scene = create_diffuse_scene(
        tilt_angle=tilt_angle,
        solar_spectrum_function=solar_spectrum_function,
        include_dye=include_dye,
    )
    return _common_simulation_runner(scene, num_photons, render, workers)


if __name__ == "__main__":
    res = run_direct_simulation(
        tilt_angle=40,
        solar_elevation=50,
        solar_azimuth=180,
        add_bottom_PV=True,
        render=True,
        add_side_PV=True,
        num_photons=100,
    )

    bottom = res[-2]
    side = res[-1]
    print(f"SIMU ENDED - bottom={bottom} side={side}")

    # run_direct_simulation(tilt_angle=10, render=True, workers=1, include_dye=True)
    input()
