import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="miniplant",
    version="0.9.1",
    author="Dario Cambie",
    author_email="dario.cambie@mpikg.mpg.de",
    description="A package to run pvtrace simulations on LSC-PMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dcambie/LSC-PM_solar_miniplant",
    packages=setuptools.find_packages(),
    install_requires=['pvtrace>=2.0.0', 'pvlib>=0.7.2', 'solcore>=5.7.1'],
    python_requires='>=3.6',
)
