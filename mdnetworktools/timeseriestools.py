"""
    Collection of jit-compiled functions to compute contacts,
    correlations and covariances.
"""

import numpy as np
import math
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

