from setuptools import find_packages, setup

with open('README.md', 'r') as readme:
    long_description = readme.read()

setup(
    name='dm',
    package_dir={"": "src"},
    packages=find_packages('src'),
    version='0.1.0-dev0',
    description='Demographic modeling module for ArcGIS Python API',
    long_description=long_description,
    author='Joel McCune',
    license='No license file',
)