from setuptools import setup, find_packages
from argparse import ArgumentParser

def read_requirements(path: str) -> str:
  """
  Reads requirements.txt document and returns the contents.

  @params path (srt): Path to requirements.txt file.
  
  @returns (str): String containing the required packages.
  """
  with open(path, 'r', encoding='utf-8-sig') as file:
    data = file.read()
  return data

def read_readme(path: str) -> str:
  """
  Reads Readme.md document and returns the content.

  @params path (srt): Path to Readme.md file.
  
  @returns (str): String containing the Readme description.
  """
  with open(path, 'r') as file:
    data = file.read()
  return data

setup(
  name = 'neuralforge',
  version = '0.0.13',
  author = 'Eduardo Leitao da Cunha Opice Leao',
  author_email = 'eduardoleao052@gmail.com',
  maintainer = 'Eduardo Leitao da Cunha Opice Leao',
  url = 'https://github.com/eduardoleao052/Autograd-from-scratch',
  python_requires = '>=3.0',
  install_requires = read_requirements('requirements.txt').split('\n'),
  description = 'An educational framework similar to PyTorch, built to be interpretable and easy to implement.',
  license = 'MIT',
  keywords = 'autograd deep-learning machine-learning ai numpy python',
  packages = find_packages(),
  long_description = read_readme('README.md'),
  long_description_content_type='text/markdown'
)