from distutils.core import setup
from setuptools import find_packages

setup(
    name='unstable_baselines3',
    version='6.9.0',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'pettingzoo',
        'torch',
        'stable-baselines3',
    ],
    license='Liscence to Krill',
)
