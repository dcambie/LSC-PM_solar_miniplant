from datetime import datetime
import logging
from pathlib import Path

from miniplant.locations import EINDHOVEN
from miniplant.full_simulation import yearlong_simulation

logging.getLogger("trimesh").disabled = True
logging.getLogger("shapely.geos").disabled = True
logging.getLogger("pvtrace").setLevel(
    logging.WARNING
)  # use logging.DEBUG for more printouts

XP_START = EINDHOVEN.pytz.localize(datetime(2020, 7, 15, 15, 0))  # 15th July 3pm
XP_END = EINDHOVEN.pytz.localize(datetime(2020, 7, 15, 17, 30))  # 15th July 5pm

target_file = Path("./experimental_conditions_results.csv")

time_interval = (XP_START, XP_END)


if __name__ == "__main__":
    yearlong_simulation(
        tilt_angle=40,
        location=EINDHOVEN,
        workers=12,
        time_resolution=60 * 30,
        include_dye=True,
        num_photons_per_simulation=10000,
        time_range=time_interval,
        target_file=target_file,
    )
