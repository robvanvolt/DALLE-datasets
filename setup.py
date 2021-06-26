from setuptools import setup, find_packages

setup(
  name = 'dalle-datasets',
  packages = find_packages(),
  include_package_data = True,
  version = '0.1.0',
  license='MIT',
  description = 'DALL-E - Datasets',
  author = 'Robert van Volt',
  author_email = 'robvanvolt@gmail.com',
  url = 'https://github.com/robvanvolt/dalle-datasets',
  keywords = [
    'artificial intelligence',
    'big data',
    'datasets',
  ],
  install_requires=[
    'pillow',
    'regex',
    'torch',
    'torchvision',
    'WebDataset',
    'tfrecord',
    'tensorflow',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.8.8',
  ],
)