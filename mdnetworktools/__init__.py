#   This file is part of the mdnetworktools repository.
#   Copyright (C) 2020 Ivanov Lab,
#   Georgia State University (USA)
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU Lesser General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
