from setuptools import find_packages
from setuptools import setup

VERSION = "0.0.0"

setup(
    name="reef-net",
    packages=find_packages(exclude=("*_test.py",)),
    version=VERSION,
    description="Starfish object detection model for the TensorFlow save the reef dataset",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    url="https://github.com/lukewood/augmentation-aware-contrastive-learning",
    author="Luke Wood, Sandra Villamar, Harsh Thakur",
    author_email="lukewoodcs@gmail.com",
    install_requires=[
        "keras-cv @ git+https://github.com/keras-team/keras-cv",
        "black",
        "isort",
        "flake8",
        "tensorflow",
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
        "dill"
    ],
)
