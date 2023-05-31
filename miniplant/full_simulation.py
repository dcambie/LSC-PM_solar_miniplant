"""
Runs a full simulation on a year, including both diffuse and direct irradiation.
Output in Einstein absorbed.
"""

import time
import logging
from pathlib import Path
from tqdm import tqdm

# Forcing numpy to single thread results in better multiprocessing performance.
# See pvtrace issue #48
# import os
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_MAX_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
#
# # Set loggers (no output needed for progress bar to work!)
# logging.getLogger("trimesh").disabled = True
# logging.getLogger("shapely.geos").disabled = True
# logging.getLogger("pvtrace").setLevel(
#     logging.WARNING
# )  # use logging.DEBUG for more printouts


import numpy as np

from pvlib.location import Location

from miniplant.scene_creator import REACTOR_AREA_IN_M2
from miniplant.simulation_runner import run_direct_simulation, run_diffuse_simulation
from miniplant.solar_data import solar_data_for_place_and_time

logger = logging.getLogger("pvtrace").getChild("miniplant")


class PhotonFactory:
    """Create a callable sampling the current solar spectrum"""

    def __init__(self, spectrum):
        self.spectrum = spectrum

    def __call__(self, *args, **kwargs):
        return self.spectrum.sample(np.random.uniform())


def yearlong_simulation(
    tilt_angle: int,
    location: Location,
    workers: int = None,
    time_resolution: int = 1800,
    num_photons_per_simulation: int = 120,
    include_dye: bool = True,
    time_range=None,
    target_file=None,
):
    logger.info(f"Starting simulation w/ tilt angle {tilt_angle}")

    solar_data = solar_data_for_place_and_time(location, tilt_angle, time_resolution)
    if time_range:
        solar_data = solar_data.loc[time_range[0] : time_range[1]]

    def calculate_productivity_for_datapoint(df):
        """
        This function is apply()ed to the dataframe to populate it with the simulation results.
        It takes care of setting up the simulation, and fill in the relevant fields or it terminates early if the
        simulation is not deemed necessary (solar position below horizon or invalid surface fraction)
        """
        logger.info(f"Current date/time {df.name}")

        # Create a function sampling the current solar spectrum
        direct_photon_factory = PhotonFactory(df["direct_spectrum"])
        diffuse_photon_factory = PhotonFactory(df["diffuse_spectrum"])

        # Get the fraction of direct photon reacted
        df["simulation_direct"] = run_direct_simulation(
            tilt_angle=tilt_angle,
            solar_azimuth=df["azimuth"],
            solar_elevation=df["apparent_elevation"],
            solar_spectrum_function=direct_photon_factory,
            num_photons=num_photons_per_simulation,
            workers=workers,
            include_dye=include_dye,
        )
        df["direct_reacted"] = (
            df["simulation_direct"] * df["direct_irradiance"] * REACTOR_AREA_IN_M2
        )

        # Get the fraction of diffuse photon reacted
        df["simulation_diffuse"] = run_diffuse_simulation(
            tilt_angle=tilt_angle,
            solar_spectrum_function=diffuse_photon_factory,
            num_photons=num_photons_per_simulation,
            workers=workers,
            include_dye=include_dye,
        )
        df["diffuse_reacted"] = (
            df["simulation_diffuse"] * df["diffuse_irradiance"] * REACTOR_AREA_IN_M2
        )

        return df

    start_time = time.time()
    tqdm.pandas(desc=f"{location.name} {tilt_angle}deg")  # Shows nice progress bar
    results = solar_data.progress_apply(calculate_productivity_for_datapoint, axis=1)
    print(f"Simulation ended in {(time.time() - start_time) / 60:.1f} minutes!")

    # Results will be saved in the following CSV file
    if not target_file:
        target_file = Path(
            f"full_simulation_results/{location.name}/{location.name}_{tilt_angle}deg_results.csv"
        )
    target_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure folder existence
    results.to_csv(
        target_file,
        columns=(
            "apparent_elevation",
            "azimuth",
            "simulation_direct",
            "direct_reacted",
            "simulation_diffuse",
            "diffuse_reacted",
        ),
    )


if __name__ == "__main__":
    # Set loggers
    logging.getLogger("trimesh").disabled = True
    logging.getLogger("shapely.geos").disabled = True
    logging.getLogger("pvtrace").setLevel(
        logging.WARNING
    )  # use logging.DEBUG for more printouts

    from miniplant.locations import EINDHOVEN

    sim_to_run = [(EINDHOVEN, 40)]

    for sim_params in sim_to_run:
        print(f"Now simulating {sim_params[0].name} at {sim_params[1]} deg tilt angle")
        yearlong_simulation(
            tilt_angle=sim_params[1],
            location=sim_params[0],
            workers=12,
            time_resolution=60 * 30,
            include_dye=True,
        )
