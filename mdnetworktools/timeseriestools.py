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

"""
    Collection of jit-compiled functions to compute contacts,
    correlations and covariances.
"""

import numpy as np
import math
from scipy.spatial.distance import pdist, squareform
from numba import jit

# Pairwise dot products
@jit(nopython=True, cache=True)
def pw_dot(x,y):
	dps = np.zeros(shape=x.shape[0])
	for i in range(x.shape[0]):
		dps[i] = np.dot(x[i], y[i])
	return dps

# Meanfree data (x - ux)
@jit(nopython=True, cache=True)
def estimate_mf(x, m):
	mf_coms = np.zeros(shape=x.shape)
	for i in range(x.shape[0]):
		mf_coms[i] = x[i] - m[i]
	return mf_coms

# Covariance
@jit(nopython=True, cache=True)
def cov(coms, means):
	rank = coms.shape[0]
	covars = np.zeros(shape=(rank,rank))
	mf_coms = estimate_mf(coms, means)
	#mf_coms = coms
	for i in range(rank):
		resi = mf_coms[i]
		for j in range(i, rank):
			resj = mf_coms[j]
			c = np.mean(pw_dot(resi, resj))
			covars[i][j] = c
			covars[j][i] = c
	return covars

# Mean-square magnitudes
@jit(nopython=True, cache=True)
def estimate_msqmag(mf_com):
	m = np.zeros(shape=mf_com.shape[0])
	for i in range(mf_com.shape[0]):
		m[i] = math.pow(np.linalg.norm(mf_com[i]), 2)
	m = math.sqrt(np.mean(m))
	return m

# Correlation
@jit(nopython=True, cache=True)
def correl(coms, means, log=False):
	rank = coms.shape[0]
	corr = np.zeros(shape=(rank, rank))
	mf_coms = estimate_mf(coms, means)
	for i in range(rank):
		resi = mf_coms[i]
		resi_m = estimate_msqmag(resi)
		for j in range(i, rank):
			resj = mf_coms[j]
			resj_m = estimate_msqmag(resj)
			c = np.mean(pw_dot(resi, resj))
			cij = c / (resi_m * resj_m)
			if log == True:
				corr[i][j] = -np.log(abs(cij))
				corr[j][i] = -np.log(abs(cij))
			else:
				corr[i][j] = cij
				corr[j][i] = cij
	return corr

# Pairwise distances
@jit(nopython=True, cache=True)
def ndist(coords):
	_dim = coords.shape[0]
	dists = np.zeros(shape=(_dim, _dim))
	for i in range(_dim-1):
		a = coords[i]
		b = coords[i+1:]
		d = np.sqrt(np.sum((b-a)**2, axis=1))
		dists[i][i+1:] = d
	return dists + dists.T

# Filter contacts by a cutoff distance
@jit(nopython=True, cache=True)
def filter_(arr, cutoff):
	_dim = arr.shape[0]
	for i in range(_dim):
		for j in range(_dim):
			if arr[i][j] > cutoff:
				arr[i][j] = 0.0
			if arr[i][j] != 0.0:
				arr[i][j] = 1.0

# Contacts
@jit(nopython=True, cache=True)
def contacts_by_chunk(coords, cutoff=0.45):
	_dim = coords.shape[0]
	contacts = np.zeros(shape=(coords.shape[1], coords.shape[1]))
	for i in range(_dim):
		frame = coords[i]
		dists = ndist(frame)
		filter_(dists, cutoff)
		contacts += dists
	return contacts

# SCIPY distance
def scipy_dist(avg_coords):
	dist_matrix = pdist(avg_coords, metric='euclidean')
	dist_matrix = squareform(dist_matrix)
	dist_matrix = dist_matrix * 10 # convert from nm to angstroms
	return dist_matrix

# Reduction from all-atom to residue level
def _squeeze(matrixA, matrixB, residues, use_min=True): 
        rank = len(residues)
        for r in range(rank-1):
            res1 = residues[r]
            for j in range(r+1, rank):
                res2 = residues[j]
		if use_min == True:
                	min_d = np.min(np.ravel(matrixA[res1][:, res2]))
                	matrixB[r][j] = min_d
                	matrixB[j][r] = min_d
		else:  # By contact
                        ones = np.where(matrixA[res1][:, res2] == 1.0)[0]
                        if len(ones) != 0:
                                matrixB[r][j] += 1.0
                                matrixB[j][r] += 1.0
			
# Slower version for larger systems (>16,000 atoms)
@jit(nopython=True, cache=True)
def _minwdist(c1, c2):
	min_d = 1000.0
	for x in range(c1.shape[0]):
		d = np.sqrt(np.sum((c2-c1[x])**2, axis=1))
		min_d = np.min(d)
	return min_d

def contacts_by_frame(frame, residues, c):
	for i in range(len(residues)-1):
		res1 = frame[residues[i]]
		for j in range(i+1, len(residues)):
			res2 = frame[residues[j]]
			min_d = _minwdist(res1, res2)
			if min_d <= 0.45:
				c[i][j] += 1.0
				c[j][i] += 1.0



