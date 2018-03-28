from setuptools import setup
from setuptools import find_packages

setup(
   name='torch_rl',
   version='0.1.3',
   description='Reinforcement learning framework based on pytorch.',
   author='Marin Vlastelica',
   author_email=None,
   packages=find_packages(),
   install_requires=['torch', 'numpy', 'pandas', 'nengo', 'nengolib', 'mpi4py'],
)