from setuptools import find_packages
from setuptools import setup

VERSION = "0.0.0"

setup(
    name="reef-net",
    packages=find_packages(exclude=("*_test.py",)),
    version=VERSION,
    description="Starfish object detection model for the COTS starfish dataset",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    url="https://github.com/lukewood/reef-net",
    author="Luke Wood, Sandra Villamar, Harsh Thakur",
    author_email="lukewoodcs@gmail.com",
    install_requires=[
        "keras-cv==0.2.6",
        "black",
        "isort",
        "flake8",
        "tensorflow>=2.9.0",
        "absl-py",
        "tensorflow_datasets",
        "ml_collections",
        "opencv-python",
        "pandas",
        "pyyaml",
        "tqdm",
        "contextlib2",
        "matplotlib",
        "click",
        "wandb",
        "tensorflow-metadata",
        "dill",
    ],
)
