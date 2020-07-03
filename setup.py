#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="zeroshoteval",
    version="0.1.0",
    author="ZSLresearch team"
    url="https://github.com/ZSLresearch-team/ZeroShotEval/",
    description="Python zero-shot learning toolkit",
    install_requires=[
        "yacs>=0.1.6",
        "matplotlib",
        "tqdm>4.29.0",
        "numpy",
        "sklearn",
        "scipy",
        "fvcore",
        "easydict",
        "tensorboard",
        "pandas",
    ],
    extras_require={
        "dev": [
            "flake8==3.8.3",
            "isort",
            "black",
            "flake8-bugbear",
            "flake8-comprehensions",
        ],
    },
    packages=find_packages(exclude=("configs")),
)
