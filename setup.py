import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="miniplant",
    version="0.9.2",
    author="Dario Cambie",
    author_email="dario.cambie@mpikg.mpg.de",
    description="A package to run pvtrace simulations on LSC-PMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dcambie/LSC-PM_solar_miniplant",
    packages=setuptools.find_packages(),
    install_requires=['pvlib>=0.8.1', 'tables', 'tqdm>=4.9.0','meshcat>=0.1.1',
                      'pvtrace @ git+https://github.com/danieljfarrell/pvtrace.git@cli'],
    python_requires='>=3.7',
    package_data={'miniplant': ['*.tsv']},
    zip_safe=True
)
