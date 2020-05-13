from distutils.core import setup

# from setuptools import setup

setup(
    name="LADiM2",
    version="2",
    description="Lagrangian Advection and Diffusion Model",
    author="Bjørn Ådlandsvik",
    author_email="bjorn@imr.no",
    packages=["ladim2"],
    requires=["numpy", "yaml", "netCDF4", "pandas"],
)
