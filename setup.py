from setuptools import find_packages, setup

with open('README.md', 'r') as readme:
    long_description = readme.read()

setup(
    name='demographic-modeling',
    package_dir={"": "src"},
    packages=find_packages('src',  include=['modeling']),
    version='0.1.0-dev0',
    description='Demographic Modeling',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Joel McCune',
    license='Apache 2.0',
)
