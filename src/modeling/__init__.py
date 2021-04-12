__title__ = 'demographic-modeling-module'
__version__ = '0.3.0'
__author__ = 'Joel McCune'
__license__ = 'Apache 2.0'
__copyright__ = 'Copyright 2020 by Joel McCune'
"""
The modeling module provides capabilities supporting quantitative geographic analysis.
"""
from .country import Country, get_countries
from .accessor import ModelingAccessor, Business, get_travel_modes

__all__ = ['get_countries', 'Country', 'Business', 'ModelingAccessor', 'get_travel_modes']
