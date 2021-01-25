

from numba import jit
import numpy as np
import math

"""Collection of functions for computing distances in batches. These need to be
invoked when the mdnetworktools.builds.Topology.natoms > 15,000.
"""
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

def _group(residues, batchsize, num=0):
        group = []
        count = 0
        for r in range(num, len(residues)):
                count += len(residues[r])
                group.append(residues[r])
                if count > batchsize:
                        break

        return np.hstack(group), r

def gen_batches(residues, batchsize):
        nbatches = int(math.ceil(len(np.hstack(residues))/float(batchsize)))
        batches = []
        ids = []
        count = 0
        n = 0
        while count < nbatches:
                g, z = _group(residues, batchsize, num=n)
                batches.append(g)
                ids.append([n, z])
                count += 1
                n = z + 1
        return batches, ids

def _reduce(batchIdX, batchIdY, val1, val2, tmp_c, residues, c, cutoff):
        for i in range(batchIdX[0], batchIdX[1]+1):
                res1 = np.subtract(residues[i], val1)
                for j in range(batchIdY[0], batchIdY[1]+1):
                        res2 = np.subtract(residues[j], val2)
                        if len(np.where(tmp_c[res1][:,res2] < cutoff)[0]) != 0:
                                min_d = np.min(np.ravel(tmp_c[res1][:,res2]))
                                c[i][j] = min_d
                                c[j][i] = min_d

def batch_distances(residues, batch, c, cutoff=0.45):
        #batches, batchIds = gen_batches(residues, batchsize)
        batches = batch[0]
        batchID = batch[1]
        for i, x_slice in enumerate(batches):
                for j, y_slice in enumerate(batches):
                        if i == j or i < j:
                                tmp_c = nppdist(coords[x_slice], coords[y_slice])
                                min1 = np.min(batches[i])
                                min2 = np.min(batches[j])
                                _reduce(batchIds[i], batchIds[j],
                                        min1, min2, tmp_c, residues,
                                        c, cutoff=cutoff)
