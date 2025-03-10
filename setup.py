from setuptools import setup, find_packages


setup(
    name="laplacian_dual_dynamics",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
