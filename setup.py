from setuptools import setup, find_packages


setup(
    name="laplacian_dual_dynamics",
    version="0.1",
    package_dir={"laplacian_dual_dynamics": "src"},
    packages=["laplacian_dual_dynamics"] + ["laplacian_dual_dynamics." + pkg for pkg in find_packages(where="src")],
)
