﻿from setuptools import setup, find_packages

def read_file(path):
  with open(path) as file:
    data = file.read()
  return data

setup(
  name = 'neuralforge',
  version = '0.0.1',
  author = 'Eduardo Leitao da Cunha Opice Leao',
  author_email = 'eduardoleao052@gmail.com',
  maintainer = 'Eduardo Leitao da Cunha Opice Leao',
  url = 'https://github.com/eduardoleao052/Autograd-from-scratch',
  python_requires = '>=3.0',
  install_requires = read_file('requirements.txt').split('\n'),
  description = 'An educational framework similar to PyTorch, built to be interpretable and easy to implement.',
  license = 'MIT',
  keywords = 'autograd deep-learning machine-learning ai numpy python',
  packages = find_packages(),
  long_description = read_file('README.md'),
  long_description_content_type='text/markdown'
)