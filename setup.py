from distutils.core import setup
from setuptools import find_packages

setup(
    name='unstable_baselines3',
    version='420.6.9',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'matplotlib',
        'pettingzoo',
        'torch',
        'stable-baselines3',
    ],
)
