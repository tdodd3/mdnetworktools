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

from setuptools import setup, find_packages

description = '''
Collection of functions for rapidly building, analyzing and 
visualizing networks from molecular dynamics simulation data.
'''

setup(
    use_scm_version=dict(root='..', relative_to=__file__),
    name='mdnetworktools',
    author='Thomas Dodd',
    author_email='tdodd224@gmail.com',
    url='https://github.com/tdodd3/mdnetworktools/',
    description=description,
    packages=find_packages(),
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
    install_requires=[
        'numpy',
        'scipy'],
    zip_safe=False)
