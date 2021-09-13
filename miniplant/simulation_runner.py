"""
Module to set up a scene to run a simulation in direct or diffuse conditions
"""
import logging
import collections
import numpy as np
import matplotlib.pyplot as plt

from typing import Callable

from pvtrace import photon_tracer, MeshcatRenderer, Event, Scene

from miniplant.scene_creator import create_direct_scene, emission

logger = logging.getLogger("pvtrace").getChild("miniplant")
logging.getLogger("matplotlib").setLevel(logging.WARNING)


def _common_simulation_runner(
    scene: Scene, num_photons: int = 100, render: bool = False, workers: int = None
):
    logger.debug(f"Starting ray-tracing with {num_photons} photons (Render is {render})")

    if render and workers > 1:
        raise RuntimeError("Sorry, cannot use renderer if more than 1 worker is used!")

    # SINGLE-THREADED
    if workers == 1:
        if render:
            renderer = MeshcatRenderer(open_browser=True)
            renderer.render(scene)
        finals = []
        led_wavelengths = []
        valid_photon = 0
        missed_photon = 0

        while True:
            ray = next(scene.emit(1))
            steps = photon_tracer.follow(scene, ray)
            path, events = zip(*steps)


            # Ensures only photons at least hitting the reactor are taken into account
            if len(events) > 2:
                valid_photon += 1
                # print(ray.wavelength)
                finals.append(events[-1])
                led_wavelengths.append(ray.wavelength)
                if render:
                    renderer.add_ray_path(path)
                if np.mod(100*valid_photon/num_photons, 10) == 0:
                    print(f"Currently {100*valid_photon/num_photons:.2f}% has been simulated")
            else:
                missed_photon += 1

            if valid_photon >= num_photons:
                plt.hist(led_wavelengths, bins=100, density=True, histtype='step',
                         label='sample')
                x = emission[:, 0]
                y = emission[:, 1]
                plt.plot(x, y / np.trapz(y, x), label='distribution')
                plt.legend()
                plt.xlabel("Wavelength (nm)")
                plt.grid(linestyle='dotted')
                plt.show()
                break

    else:
        if render:
            logger.warning(
                "Cannot use renderer in multi-threaded simulations! Set workers to 1 for single-theaded"
            )
        # MULTI-THREADED
        results = scene.simulate(num_rays=num_photons, workers=workers)
        finals = [photon[-1][1] for photon in results]

    count_events = collections.Counter(finals)
    reacted_fraction = count_events[Event.REACT] / valid_photon
    # print(count_events[Event.KILL])
    # print(count_events[Event.REACT])
    # print(count_events[Event.EMIT])
    # print(count_events[Event.ABSORB])
    # print(count_events[Event.EXIT])
    # print(count_events[Event.REFLECT])
    # print(count_events[Event.NONRADIATIVE])
    # print(count_events[Event.SCATTER])
    # print(count_events[Event.TRANSMIT])
    total_photons = valid_photon + missed_photon
    # print(total_photons)
    # print(valid_photon)
    # print(missed_photon)
    missed_fraction = missed_photon / total_photons
    logger.debug(f"Fraction of photons missed was {missed_fraction:.3f}")
    logger.debug(f"*** SIMULATION ENDED *** (Efficiency was {reacted_fraction:.3f})")

    return reacted_fraction


def run_direct_simulation(
    light_distribution: Callable = lambda: 555,
    num_photons: int = 100,
    render: bool = False,
    workers: int = None,
    include_dye: bool = None
) -> float:
    """
    Create a scene for direct irradiation with the provided parameters and runs a simulation on it
    """
    scene = create_direct_scene(include_dye=include_dye)
    return _common_simulation_runner(scene, num_photons, render, workers)


if __name__ == "__main__":
    run_direct_simulation(num_photons=10000, render=True, workers=1, include_dye=False)
    input()
