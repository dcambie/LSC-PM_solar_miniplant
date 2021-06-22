from datetime import datetime, timezone
import logging
from pathlib import Path

from miniplant.full_simulation import yearlong_simulation

logging.getLogger("trimesh").disabled = True
logging.getLogger("shapely.geos").disabled = True
logging.getLogger("pvtrace").setLevel(logging.WARNING)  # use logging.DEBUG for more printouts

from miniplant.locations import EINDHOVEN

XP_START = datetime(2020, 7, 15, 15, 0, tzinfo=EINDHOVEN.pytz)  # 15th July 3pm
XP_END = datetime(2020, 7, 15, 17, 0, tzinfo=EINDHOVEN.pytz)  # 15th July 5pm

target_file = Path(f"./experimental_conditions_results.csv")

time_interval = (XP_START, XP_END)


if __name__ == '__main__':
    yearlong_simulation(tilt_angle=40, location=EINDHOVEN, workers=12, time_resolution=60 * 30, include_dye=True,
                        num_photons_per_simulation=10000, time_range=time_interval, target_file=target_file)
