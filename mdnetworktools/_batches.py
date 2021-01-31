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

"""Collection of functions for computing distances in batches. These need to be
invoked when the mdnetworktools.builds.Topology.natoms > 15,000.
"""

from numba import jit
import numpy as np
import math


# Pairwise distances between 2 sets of coordinates
@jit(nopython=True, cache=True)
def nppdist(coords1, coords2):
        d1 = coords1.shape[0]
        d2 = coords2.shape[0]
        dists = np.zeros(shape=(d1, d2))
        for i in range(d1):
                a = coords1[i]
                b = coords2
                d = np.sqrt(np.sum((b-a)**2, axis=1))
                dists[i][:] = d
        return dists

# Grouping atoms by residue into batch of size == batchsize
def _group(residues, batchsize, num=0):
        group = []
        count = 0
        for r in range(num, len(residues)):
                count += len(residues[r])
                group.append(residues[r])
                if count > batchsize:
                        break
        return np.hstack(group), r

# Generates batches of size==batchsize for the given list of residues
def gen_batches(residues, batchsize):
        nbatches = int(math.ceil(len(np.hstack(residues))/float(batchsize)))
        batches = []
        ids = []
        count = 0 
        n = 0 # Place holder for resID
        while count < nbatches:
                g, z = _group(residues, batchsize, num=n)
                batches.append(g)
                ids.append([n, z])
                count += 1
                n = z + 1
        return batches, ids

# Reduction from all-atom distances to closest-heavy between residues
def _reduce(batchIdX, batchIdY, val1, val2, tmp_c, residues, c):
        """
        Parameters
        ------------
        batchIdx : list, residue IDs in batchX
        batchIdy : list, residue IDs in batchY
        val1 : int, offset to index in subarray tmp_c
        val2 : int, offset to index in subarray tmp_c
        tmp_c : array, subarray of all-atom distances
        residues : list, atoms grouped by residue
        c : array, residue distance matrix to be modified in-place
        
        Returns
        ------------
        None - distance matrix is modified in-place
        
        """
        
        for i in range(batchIdX[0], batchIdX[1]+1):
                res1 = np.subtract(residues[i], val1) # Compute index to tmp_c
                for j in range(batchIdY[0], batchIdY[1]+1):
                        res2 = np.subtract(residues[j], val2) # Compute index to tmp_c
                        min_d = np.min(np.ravel(tmp_c[res1][:,res2]))
                        c[i][j] = min_d
                        c[j][i] = min_d

# Computes distances in batches                                
def batch_distances(residues, batch, coords, c):
        """
        Parameters
        ------------
        residues : list
        batch : tuple, where batch[0] are the batches and batch[1] are the batch IDs
        coords : array
        c : array, residue distance matrix to be modified in-place
        
        Returns
        ------------
        None - Distance matrix c is modified in-place
        
        Note:
                We are still dealing with a symmetric matrix (at least, it will be).
                Therefore, we only need to compute distances between each batch with
                itself and each batch with every other batch once. For example, if there are 3 
                batches then this function will compute distances in the following order:
                (0,0), (0,1), (0,2), (1,1), (1,2), (2,2). 
        """
        
        batches = batch[0]
        batchIds = batch[1]
        for i, x_slice in enumerate(batches):
                for j, y_slice in enumerate(batches):
                        if i == j or i < j:
                                tmp_c = nppdist(coords[x_slice], coords[y_slice])
                                min1 = np.min(batches[i])
                                min2 = np.min(batches[j])
                                _reduce(batchIds[i], batchIds[j],
                                        min1, min2, tmp_c, residues, c)
                                
# Find nonzero elements by residue                                
def gen_nonzero(c, cutoff):
        w = []
        for x in range(c.shape[0]):
                j = c[:,x]
                s = np.where(j <= cutoff)[0]
                w.append(np.asarray([y for y in s if y != x]))
        return w

def _reduce2(residue1, residues2, ind1, w, distarr, c, use_min=True):
        """Reduce subarray of all-atom distances to residue-level
        contacts. The contact map, c, is modified in-place.
        """
        
        # residue1 is dim1 and residues2 is dim2
        res1 = np.subtract(residue1, np.min(residue1)) # indices in subarray
        count = 0
        for i in range(len(residues2)):
                ind2 = w[i] # index in contact map c
                r_b = residues2[i]
                # Indices of residue2 in subarray
                res2 = [j+count for j in range(len(r_b))]
                if use_min == True:
                        min_d = np.min(distarr[res1][:, res2])
                        if min_d <= 0.45:
                                c[ind1][ind2] += 1.0
                else: # For CUDA version
                        maxc = np.max(distarr[res1][:, res2])
                        if maxc != 0.0:
                                c[ind1][ind2] += maxc
                count += len(r_b)

# Distances for a subset that has been pre-determined
def _accumulate(w, coords, residues, c, enable_cuda=False):
        """Determine distances from subset of batches and populate
        the contact matrix with 0 or 1.
        
        Parameters
        ------------
        w : list, Residues that have nonzero elements in the full distance array
        coords : array, md.Trajectory.Frame reference
        residues : list, list-of-lists containing atom indices by residue
        c : array, contact map by residue
        enable_cuda : bool, use an available GPU to compute contacts
        
        Returns
        ------------
        None - contact map is modified in-place
        """
        if enable_cuda == True:
                import utilCUDA as uc
        for i in range(len(w)):
                subset = [residues[x] for x in w[i]]
                batch1 = residues[i]
                batch2 = np.hstack(subset)
                if enable_cuda == True:
                        d = uc.batch_pwc_CUDA(coords[batch1], coords[batch2])
                        _reduce2(batch1, subset, i, w[i], d, c, use_min=False)
                else:
                        d = nppdist(coords[batch1], coords[batch2])
                        _reduce2(batch1, subset, i, w[i], d, c)
