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

import numpy as np
import mdtraj as md
import timeseriestools as tst
from scipy.spatial.distance import pdist, squareform

class Topology(object):
    def __init__(self, topFile):
        self.topFile = topFile

    def init_top(self):
        _masses = {'C':12.01, 'O':16.00,
                 'N':14.00, 'S':32.07,
                 'P':30.97}
        topology = md.load_prmtop(self.topFile)
        ind1 = topology.select("protein")
        ind2 = topology.select("name P or name OP1 or name OP2"+ \
                               " or name N1 or name N3")
        tmp_ind = np.concatenate((ind1, ind2))
        table, bonds = topology.to_dataframe()
        rtop = {}
        indices = []
        loc_index = 0 # new atom index mapping once trajectory is processed
        for r in tmp_ind:
            res = table['resSeq'][r]
            atom = table['element'][r]
            if atom in _masses:
                if res not in rtop:
                   rtop[res] = {loc_index: _masses[atom]}
                else:
                   rtop[res][loc_index] = _masses[atom]
                loc_index += 1
                indices.append(r)
        #self.rtop = rtop
        #self.indices = indices
        return rtop, indices

class DynamicNetwork(Topology):
    def __init__(self, topFile, trajFile, ref=None):
        super(DynamicNetwork, self).__init__(topFile)
        self.rtop, self.indices = self.init_top()
        self.trajFile = trajFile
        self.chunks = []
        # When ref != None, this indicates that the trajectory
        # has not been aligned and will be prior to the 
        # calculation of the covariance or correlation matrix
        if ref is not None:
            self.ref = md.load_pdb(ref)
            self.ref_indices = topology.select('name CA or name P')

    def com_coords(self, coords):
        com_coords = []
        com_means = []
        _len = coords.shape[0]
        residues = list(self.rtop.keys())
        rank = len(residues)
        for i in range(rank):
                atoms = list(self.rtop[i].keys())
                weights = np.asarray(list(self.rtop[i].values()))
                c = (np.asarray([coords[:,atoms][:,i]*weights[i] \
                        for i in range(len(weights))]).sum(axis=0))/np.sum(weights)
                com_means.append(np.mean(c, axis=0))
                com_coords.append(c)
        return np.asarray(com_coords), np.asarray(com_means)


    def process_input(self, chunk_size=100, stride=1, align=False):
        """Process the input trajectory and store the chunks
           for future use

        Parameters
        ------------
        chunk_size : int
            Number of frames to process at one time
        stride : int
            Number of frames to skip when reading
        align : bool
           Set to True to align the MD trajectory to a reference
           using mdtraj.Trajectory.superpose 
        
        Returns
        ------------
        trajectory coordinates : list of NumPy arrays containing COM coordinates
             where is array in the list is of shape [chunk_size, n_residues, 3]
             
        """
        #raw_chunks = []
        com_data = []
        chunk_weights = []
        for chunk in md.iterload(self.trajFile, top=self.topFile,
                                 chunk=chunk_size, stride=stride, 
                                 atom_indices=self.indices):
            if align == True:
                chunk.superpose(self.ref, atom_indices=self.ref_indices,
                                ref_atom_indices=self.ref_indices)
            coords = chunk.xyz
            #raw_chunks.append(coords)
            com_data.append(self.com_coords(coords))
            chunk_weights.append(coords.shape[0])
        #self.chunks = raw_chunks
        self.com_data = com_data
        chunk_weights = np.asarray(chunk_weights)
        self.chunk_weights = chunk_weights / float(np.max(chunk_weights))
        self.chunk_size = chunk_size
        self.stride = stride
        
    def compute_avg_dist_matrix(self):
        """Compute the average distance between selected residues
            in a given MD trajectory.

        Note that a call to process_ must happen before this method
        can be used (e.g. self.chunks must exist!)
        
        Returns
        ------------
        average distances
            NumPy array of NxN, where N == number of selected indices
    
        """
        residues = list(self.rtop.keys())
        rank = len(residues)
        avg_dists = None
        iter_ = 0
        #for coords in self.chunks:
        for chunk in md.iterload(self.trajFile, top=self.topFile,
                                 chunk=self.chunk_size, stride=self.stride,
                                 atom_indices=self.indices):

            coords = chunk.xyz
            sum_coords = np.sum(coords, axis=0)
            if avg_dists is None:
                avg_dists = sum_coords
            else:
                avg_dists += sum_coords
            iter_ += coords.shape[0]
        avg_dists = avg_dists / float(iter_)
        dist_matrix = pdist(avg_dists, metric='euclidean')
        dist_matrix = squareform(dist_matrix)
        dist_matrix = dist_matrix * 10 # convert from nm to angstroms
        
        # Find the closest distance between heavy atoms in each 
        # residue pair - this is currently the bottleneck.
        
        avg_dist_matrix = np.zeros(shape=(rank,rank))
        for r in range(rank-1):
            res1 = list(self.rtop[residues[r]].keys())
            for j in range(r+1, rank):
                res2 = list(self.rtop[residues[j]].keys())
                min_d = np.min(np.ravel(dist_matrix[res1][:, res2]))
                avg_dist_matrix[r][j] = min_d
                avg_dist_matrix[j][r] = min_d
       
        return avg_dist_matrix 

    def compute_covariance_matrix(self, enable_cuda=False):
        """Computes the covariance between all residues in the processed
           chunks from the input MD trajectory

        A call to process_input must happen before this method is available.
        
        Parameters
        ------------
        enable_cuda : bool
            Indicates whether to offload covariance calculation onto an
            available GPU. Default value is False.
            
        Returns
        ------------
        covariance matrix : NumPy array of shape [n_residues, n_residues]
     
        """
        
        if enable_cuda == True:
           import utilCUDA as uc

        rank = len(list(self.rtop.keys()))
        self.cov = np.zeros(shape=(rank, rank))
        iter_ = 0
        w_sum = 0
        for chunk in self.com_data:
            if enable_cuda == True:
               cov = uc.compute_covariances_CUDA(chunk)
            else:
               cov = tst.cov(chunk[0], chunk[1])
            self.cov += cov
            w_sum += (1*self.chunk_weights[iter_])
            iter_ += 1
        self.cov /= w_sum

    def compute_correlation_matrix(self, log=False, enable_cuda=False):
        """Computes the correlation between all residues in the processed
           chunks of the input MD trajectory
   
        A call to process_input must happen before this method is available.
       
        Parameters
        -------------
        log : bool
         Indicates if the correlations should be transformed to weights
         using wij = -np.log(abs(cij)). Default value is False 

        enable_cuda : bool
         Indicates whether to offload correlation calculation onto an 
         available GPU. Default value is False

        Returns
        -------------
         correlation matrix : NumPy array of shape [n_residues, n_residues]

        """
        
        if enable_cuda == True:
           import utilCUDA as uc

        rank = len(list(self.rtop.keys()))
        self.corr = np.zeros(shape=(rank, rank))
        iter_ = 0
        w_sum = 0
        for chunk in self.com_data:
            if enable_cuda == True:
               corr = uc.compute_correlations_CUDA(chunk)
            else:
               corr = tst.correl(chunk[0], chunk[1], log=log)
            self.corr += corr
            w_sum += (1*self.chunk_weights[iter_])
            iter_ += 1
        self.corr /= w_sum

        # Due to rounding errors correlations tend to fall outside 
        # the range of -1, 1. This is the easiest work around.
        np.clip(self.corr.real, -1, 1, out=self.corr.real)

        if enable_cuda == True and log == True:
           self.corr = -np.log(np.abs(self.corr))
           self.corr[self.corr == np.inf] = 0.0

    def postprocess(self, scheme='i+2'):
        """Remove correlations that correspond to the scheme
           given. Possible schemes are only i+n, where n=1,2,3...
 
        Parameters
        -------------
        scheme : string
              Determines where correlations should be set to zero.
              For example, i+2 indicates that residues i+1 and i+2
              should be set to zero, where i == current residue.

        Returns
        -------------
        modified correlation matrix : NumPy array of shape [n_residues, n_residues]

        """
        interval = int(scheme.split('+')[-1])
        rank = len(list(self.rtop.keys()))
        for i in range(rank-interval):
            ind = [i+j+1 for j in range(interval)]
            for k in ind:
                self.corr[i][k] = 0.0
                self.corr[k][i] = 0.0

    def build_network(self, cutoff=4.5, scheme='all', enable_cuda=False):
        self.dist_matrix = self.compute_avg_dist_matrix()
        self.compute_correlation_matrix(log=True, enable_cuda=enable_cuda)
        
        np.savetxt("dist.dat", self.dist_matrix, fmt="%0.5f")
        np.savetxt("corr.dat", self.corr, fmt="%0.5f")
        
        # Modify the distance matrix to include only distances
        # within the cutoff
        self.dist_matrix[self.dist_matrix > cutoff] = 0.0
        self.dist_matrix[self.dist_matrix != 0.0] = 1.0
        np.savetxt("contactmap.dat", self.dist_matrix, fmt="%0.5f")
        
        if scheme != 'all':
           self.postprocess(scheme=scheme)
        self.network = self.dist_matrix * self.corr
        np.savetxt("network.dat", self.network, fmt="%0.5f")


class DifferenceNetwork(Topology):
    def __init__(self, topFiles, trajFiles):
        self.topFiles = topFiles
        self.trajs = trajFiles
        self.rtops = []
        self.indices = []
        for top in topFiles:
            super(DifferenceNetwork, self).__init__(top)
            r, i = self.init_top()
            self.rtops.append(r)
            self.indices.append(i)

    def process_traj(self, traj, top, indices, chunk_size=100, stride=1):
        """Process the input trajectory and store the chunks
           for future use

        Parameters
        ------------
        traj : str
            Path to MD trajectory
        top : str
            Path to topology file 
        indices : array-like
            Atom indices to load from MD trajectory
        chunk_size : int
            Number of frames to process at one time
        stride : int
            Number of frames to skip when reading

        Returns
        ------------
        trajectory coordinates : list of NumPy arrays
             where is array in the list is of shape [chunk_size, self.indices, 3]

        """
        raw_chunks = []
        chunk_weights = []
        for chunk in md.iterload(traj, top=top, chunk=chunk_size, 
                                 stride=stride, atom_indices=indices):

            coords = chunk.xyz
            raw_chunks.append(coords)
            chunk_weights.append(coords.shape[0])
    
        return raw_chunks, chunk_weights

    def compute_contacts(self, chunks, rtop, indices):
        """Computes contacts between all residues in reduced topology.
           Contacts between residues are defined as being within 4.5 angstroms
           of each other.

        Parameters
        -------------
        chunks : list of NumPy arrays
               Each array contains the coordinates from a chunk of the MD
               trajectory and is of shape [n_frames, n_atoms, 3]
        rtop : Topology.rtop object
        indices : array-like
               Array of atom indices 

        Returns
        -------------
        contact map : NumPy array
               N x N matrix containing contact probabilities between all residues
        
        """
        residues = [list(rtop[i].keys()) for i in rtop]
        res_pairs = [[residues[i], residues[j]] for i in range(len(residues)-1) \
                                                for j in range(i+1, len(residues))]

        rank = len(indices)
        tframes = sum([i.shape[0] for i in chunks])
        c = np.zeros(shape=(rank, rank))

        for chunk in chunks:
            tmp_c = tst.contacts_by_chunk(chunk)
            c += tmp_c
        c /= tframes
        
        # Reduce the all-atom contact map to just the largest probable contact
        # between residues
        contacts = np.zeros(shape=(len(residues),len(residues)))
        
        for p in res_pairs:
            cprobij = np.max(np.ravel(c[p[0]][:,p[1]]))
            contacts[p[0]][p[1]] = cprobij
            contacts[p[1]][p[0]] = cprobij
        
        return contacts

    def consensus_network(self, states, cutoff):
        consensus_matrix = np.zeros(shape=states[0].shape)
        
        for state in states:
            consensus_matrix += state

        # Normalize matrix
        consensus_matrix /= np.max(consensus_matrix)
        zeros = np.where(consensus_matrix <= cutoff)
        ones = np.where(consensus_matrix >= cutoff)
        
        # Remove weights and assign 1 and 0 accordingly
        consensus_matrix[zeros] = 0.0
        consensus_matrix[ones] = 1.0
        
        return consensus_matrix

    def diff_network(self, states):
        diff = states[0]
        
        for i in range(1, len(states)):
            diff -= states[i]
        
        return diff

    def build_network(self, cutoff=0.90, chunk_size=100, stride=1):
        states = []
        
        for i in range(len(self.topFiles)):
            current_indices = np.asarray(self.indices[i])
            chunks, _ = self.process_traj(self.trajs[i], self.topFiles[i],
                                          current_indices, chunk_size=chunk_size,
                                          stride=stride)
            state = self.compute_contacts(chunks, self.rtops[i], current_indices)
            states.append(state)
        
        self.consensus_matrix = self.consensus_network(states, cutoff=cutoff)
        self.difference_matrix = self.diff_network(states)
        
        np.savetxt("consensus.dat", self.consensus_matrix, fmt="%0.0f")
        np.savetxt("difference.dat", self.difference_matrix, fmt="%0.5f")
                
    def get_communities(self, commFile, offset):
        s = {}
        count = 0
        with open(commFile) as f:
             for line in f:
                 lines = line.split()
                 resids = [int(r)+offset for r in lines]
                 s[count] = resids
                 count += 1
        return s

    def deltaP(self, commFile, diff, offset=0):
        comms = self.get_communities(commFile)
        nonzero = np.where(diff != 0.0)
        contacts = {}

        for c in range(nonzero[0].shape[0]):
            ind1 = nonzero[0][c]
            ind2 = nonzero[1][c]
            p = diff[ind1][ind2]
            cc = []
            for com in comms:
                if ind1 in comms[com] and ind2 not in comms[com]:
                   cc.append(com)
                if ind2 in comms[com] and ind1 not in comms[com]:
                   cc.append(com)
            cc = tuple(cc)
            if cc not in contacts and len(cc) > 1:
               contacts[cc] = p
            if cc in contacts and len(cc) > 1:
               contacts[cc] += p

        return contacts
 
    def write_deltaP(self, deltaP, oname="deltaP.dat"):
        f = open(oname, "w")
        for pair in deltaP:
            f.write("{} {} {}\n".format(pair[0], pair[1], deltaP[pair]))
        f.close()

