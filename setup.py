from setuptools import setup

# from setuptools import setup

setup(
    name="LADiM2",
    version="2",
    description="Lagrangian Advection and Diffusion Model",
    author="Bjørn Ådlandsvik",
    author_email="bjorn@imr.no",
    packages=["ladim2"],
    entry_points={'console_scripts': ['ladim2=ladim2.main:script']},
    install_requires=["numpy", "pyyaml", "netCDF4", "pandas"],
)
