from miniplant.solar_data import solar_data_for_place_and_time
from miniplant.locations import LOCATIONS


def test_solar_data_for_place_and_time():
    tilt = {
        "Townsville": -10,
        "Eindhoven": 40,
        "Plataforma Solar de Almería": 30,
        "North Cape": 50
    }
    points = {
        "Townsville": 366,
        "Eindhoven": 366,
        "Plataforma Solar de Almería": 366,
        "North Cape": 298
    }

    for site in LOCATIONS:
        # Calculate solar spectrum and position per every time point
        test_df = solar_data_for_place_and_time(site, tilt[site.name], 60 * 60 * 12)

        assert test_df["azimuth"].count() == points[site.name]

