from setuptools import setup, find_packages

setup(
    name="life_sim",
    packages=find_packages(),
    version="0.0.1",
    install_requires=["gymnasium", "numpy"],
)