from setuptools import setup, find_packages  # type: ignore

setup(
    name="dataenvgym",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
)
