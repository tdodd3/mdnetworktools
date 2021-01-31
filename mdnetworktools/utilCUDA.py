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
    Collection of CUDA functions to compute contacts,
    correlations and covariances.
"""

from numba import cuda
import numpy as np
import math
import timeseriestools as tst
import os
import warnings

# Set the global constants for threads per block (TPB) and
# GPU device (CU_DEVICE). Check to see if the user overrode 
# the defaults and change if necessary.

#FILE = "numbaVAR.sh"
TPB = 512
CU_DEVICE = 0

try:
	TPB = int(os.environ['TPB'])
except:
	warnings.warn("Threads per block ($TPB) variable not defined. " + \
		      "Setting the default value to 512 threads. If " + \
		      "this is not the desired behavior, please set " + \
		      "variable manually.")
	pass

try:
	CU_DEVICE = int(os.environ['CU_DEVICE'])
except:
	warnings.warn("Device number ($CU_DEVICE) not defined. " + \
		      "Setting the default to device 0. If this " + \
		      "is not the desired behavior, please set " + \
		      "variable manually.")
	pass



#### Device functions ####

# Distance
@cuda.jit('float32(float32[:], float32[:])', device=True, inline=True)
def cdist(v1, v2):
	x = (v2[0] - v1[0])**2
	y = (v2[1] - v1[1])**2
	z = (v2[2] - v1[2])**2
	d = math.sqrt(x+y+z)
	return d

# Pairwise contacts
@cuda.jit('float32(float32[:,:], float32[:,:])', device=True, inline=True)
def pwcontacts(arr1, arr2):
        s = 0.0
        for i in range(arr1.shape[0]):
                d = cdist(arr1[i], arr2[i])
                if d <= 0.45 and d != 0.0:
                        s += 1.0
        return s

# Dot product
@cuda.jit('float32(float32[:], float32[:])', device=True, inline=True)
def dot(v1, v2):
	s = 0.0
	_dim = int(v1.shape[0])
	for x in range(_dim):
		s += (v1[x] * v2[x])
	return s

# Mean pairwise dot products
@cuda.jit('float32(float32[:,:], float32[:,:])', device=True, inline=True)
def mpw_dot(res1, res2):
	dps = 0.0
	_dim = int(res1.shape[0])
	for x in range(_dim):
		dps += dot(res1[x], res2[x])
	return dps / _dim	

# Vector magnitude
@cuda.jit('float32(float32[:])', device=True, inline=True)
def mag(v1):
	s = 0.0
	for x in v1:
		s += (x**2)
	return math.sqrt(s)

# Mean square magnitudes
@cuda.jit('float32(float32[:,:])', device=True, inline=True)
def msqm(coords):
	m = 0.0
	_dim = int(coords.shape[0])
	for x in range(_dim):
		m += (mag(coords[x]))**2
	return math.sqrt(m / _dim)


#### CUDA kernel functions ####
    
# Covariance
@cuda.jit
def cudaCovar(chunk, coords, cov):

	x = cuda.grid(1)
	tx = cuda.threadIdx.x

	if x >= cov.shape[0]:
		return 

	ref = chunk[tx]
	_dim = int(ref.shape[0])
	_len = int(coords.shape[0])
	for i in range(_dim):
		r_i = i + (_dim * tx)
		if r_i >= cov.shape[0]:
			return
		res1 = ref[i]
		for j in range(r_i, _len):
			res2 = coords[j]
			cij = mpw_dot(res1, res2)
			cov[r_i, j] = cij

# Correlation
@cuda.jit
def cudaCorrel(chunk, coords, corr):

	x = cuda.grid(1)
	tx = cuda.threadIdx.x
	
	if x >= corr.shape[0]:
		return
	
	ref = chunk[tx]
	_dim = int(ref.shape[0])
	_len = int(coords.shape[0])
	for i in range(_dim):
		r_i = i + (_dim * tx)
		if r_i >= corr.shape[0]-1:
			return
		res1 = ref[i]
		mag1 = msqm(res1)
		for j in range(r_i+1, _len):
			res2 = coords[j]
			mag2 = msqm(res2)
			c = mpw_dot(res1, res2)
			cij = c / (mag1 * mag2)
			corr[r_i, j] = cij

# Pairwise distance - while technically faster than the jit version
# this method is still inefficient and needs a future fix
@cuda.jit		
def cudaContacts(chunk, coords, dists):
	
	x = cuda.grid(1)
	tx = cuda.threadIdx.x

	if x >= dists.shape[0]:
		return
	
	ref = chunk[tx]
	_dim = int(ref.shape[0])
	_len = int(coords.shape[0])
	for i in range(_dim):
		r_i = i + (_dim * tx)
		# Check bounds of NxN array
		if r_i >= dists.shape[0]:
			return
		res1 = ref[i]	
		for j in range(r_i, _len):
			res2 = coords[j]
			c = pwcontacts(res1, res2)
			dists[r_i, j] = c

# Contacts for _accumulate_CUDA in _batches.py
@cuda.jit
def batch_contacts(ref_coords, coords, out):
        x = cuda.grid(1)
        tx = cuda.threadIdx.x

        if x > out.shape[1]:
                return

        _dim = int(ref_coords.shape[0])
        d = coords[tx]
        for i in range(_dim):
                out[i, tx] = pwcontacts(ref_coords[i], d)

#### Data prep methods ####

# Reshaping input coordinates
def _reshape(coords):
        x = [coords[:,i] for i in range(coords.shape[1])]
        return np.asarray(x, dtype=np.float32)

# Splitting input based on TPB
def split_coords(coords):
	# Determine how to split up the coordinates
	fact = int(math.ceil(float(coords.shape[0])/TPB))

	coord_split = [coords[i:i+fact] for i in range(0, coords.shape[0], fact)]

	# We need to ensure that every chunk in the coord split has the
	# same shape. If one does not then we create a zero array of the desired shape
	# and fill in the misshaped chunk. 
	# This allows us to convert the list to np array of float32 datatype
	shapeof = coord_split[0].shape
	for i in range(len(coord_split)):
		if coord_split[i].shape[0] != fact:
			tmp = np.zeros(shape=shapeof)
			tmp[:coord_split[i].shape[0]] = coord_split[i]
			coord_split[i] = tmp
	coord_split = np.asarray(coord_split, dtype=np.float32)
	return coord_split

#### Main calls used by mdnetworktools.py ####

# Contacts    
def contacts_by_frame_CUDA(coords, device=CU_DEVICE):
	cuda.select_device(device)
        natoms = int(coords.shape[1])
        r_coords = _reshape(coords)
        chunk = split_coords(r_coords)
        # Set all CUDA parameters
        A_mem = cuda.to_device(chunk)
        B_mem = cuda.to_device(r_coords)
        C_mem = cuda.device_array((natoms,natoms))
        blockspergrid = 1
        threadsperblock = TPB

        # Execute on TPB * 1 threads
        cudaContacts[blockspergrid, threadsperblock](A_mem, B_mem, C_mem)
        contacts = C_mem.copy_to_host()

        return (contacts + contacts.T)

# Correlations
def compute_correlations_CUDA(coords, device=CU_DEVICE):
	cuda.select_device(device)
	mf_coms = tst.estimate_mf(coords[0], coords[1]).astype(np.float32)
	coord_split = split_coords(mf_coms)
	A_mem = cuda.to_device(coord_split)
	B_mem = cuda.to_device(mf_coms)
	C_mem = cuda.device_array((mf_coms.shape[0], mf_coms.shape[0]))
	blockspergrid = 1
	threadsperblock = TPB
	cudaCorrel[blockspergrid, threadsperblock](A_mem, B_mem, C_mem)
	c = C_mem.copy_to_host()
	return c + c.T

# Covariances
def compute_covariances_CUDA(coords, device=CU_DEVICE):
	cuda.select_device(device)
	mf_coms = tst.estimate_mf(coords[0], coords[1]).astype(np.float32)
	coord_split = split_coords(mf_coms)
	A_mem = cuda.to_device(coord_split)
	B_mem = cuda.to_device(mf_coms)
	C_mem = cuda.device_array((mf_coms.shape[0], mf_coms.shape[0]))
	blockspergrid = 1
	threadsperblock = TPB
	cudaCovar[blockspergrid, threadsperblock](A_mem, B_mem, C_mem)
	c = C_mem.copy_to_host()
	return c + c.T

# Batch contacts used by _batches._accumulate_CUDA
def batch_pwc_CUDA(ref_c, coords, device=1):
        cuda.select_device(device)
        dim1 = ref_c.shape[0]
        dim2 = coords.shape[0]
        if dim1 > TPB:
                chunk = split_coords(coords)
                threadsperblock = TPB
        else:
                chunk = coords
                threadsperblock = coords.shape[0]
        # Set all CUDA parameters
        A_mem = cuda.to_device(ref_c)
        B_mem = cuda.to_device(chunk)
        C_mem = cuda.device_array((dim1,dim2))
        blockspergrid = 1

        # Execute on TPB * 1 threads
        batch_contacts[blockspergrid, threadsperblock](A_mem, B_mem, C_mem)
        contacts = C_mem.copy_to_host()

        return contacts
