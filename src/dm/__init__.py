__title__ = 'demographic-modeling-module'
__version__ = '0.0.0'
__author__ = 'Joel McCune'
__license__ = 'No license file'
__copyright__ = 'Copyright 2020 by Joel McCune'

# add specific imports below if you want more control over what is visible
from .country import Country
from . import utils

__all__ = [utils, Country]
