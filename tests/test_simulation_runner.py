from miniplant.simulation_runner import run_direct_simulation, run_diffuse_simulation

def green_photons():
    return 555

def red_photons():
    return 700

def test_run_direct_simulation():
    good = run_direct_simulation(tilt_angle=40, solar_elevation=50, solar_azimuth=180,
                                 solar_spectrum_function=green_photons, num_photons=200)
    assert 0.20 <= good <= 0.40
    bad = run_direct_simulation(tilt_angle=0, solar_elevation=10, solar_azimuth=180,
                                 solar_spectrum_function=red_photons, num_photons=200)
    assert bad <= 0.2


def test_run_diffuse_simulation():
    standard = run_diffuse_simulation(solar_spectrum_function=green_photons)
    assert 0.20 <= standard <= 0.40
