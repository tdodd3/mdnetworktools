# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 12:22:21 2021

@author: tdodd
"""

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = 'x.y.z'
del get_distribution, DistributionNotFound

__author__ = "Thomas Dodd"
__email__ = "tdodd224@gmail.com"

from . import mdnetworktools
from . import timeseriestools
from . import utilCUDA
from . import visualization