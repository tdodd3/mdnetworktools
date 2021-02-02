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
import math
import time
from _logger import LOG
#from scipy.spatial.distance import pdist, squareform

class Topology(object):
    """Reduced representation of the input topology
       in which the indices of the heavy atoms in the protein, or P, OP1
       OP2, N1 and N3 atoms in DNA, along with their masses are stored
       in a dictionary.
        
       Parameters
       -------------
       topFile : string
          path to the topology file
         
    """
    
    def __init__(self, topFile):
        self.topFile = topFile

    def init_top(self):
        """
        Returns
        -----------
        rtop : python dict 
           Reduced representation of the input topology
        indices : NumPy ndarray of shape [n_atoms,]
           New index mapping for each atom
        """
        
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
        
        self.natoms = len(indices)
        self.residues = [list(rtop[i].keys()) for i in rtop]
        self.nresidues = len(self.residues)
        
        return rtop, indices

class DynamicNetwork(Topology):
    """Builds the network from the input topology and trajectory
    using cross-correlation and in-contact residues. 
    
    Parameters
    ------------
    topFile : str
       Path to topology file
    trajFile : str
       Path to MD trajectory
    ref : str
       Path to reference PDB - should only be specified if the
       input trajectory needs to be aligned prior to computing 
       correlations (see below).
       
    """
    
    def __init__(self, topFile, trajFile, ref=None):
        super(DynamicNetwork, self).__init__(topFile)
        self.rtop, self.indices = self.init_top()
        self.trajFile = trajFile
        self.chunks = []
        self.log = LOG("loggy.log", overwrite=True)
        # When ref != None, this indicates that the trajectory
        # has not been aligned and will be prior to the 
        # calculation of the covariance or correlation matrix
        if ref is not None:
            self.ref = md.load_pdb(ref)
            topology = md.load(ref).topology
            self.ref_indices = topology.select('name CA or name P')
            
        self.log._startup()
        params = {"Topology": topFile, "Trajectory": trajFile, 
                  "Number of atoms": self.natoms,
                  "Number of residues": self.nresidues}
        self.log._logit((0,0), params=params)
        
        self.MEM_OK = True
        if self.natoms > 15999: # We need to invoke memory-friendly functions
            self.MEM_OK = False

    def com_coords(self, coords):
        """Computes the center of mass for each residue.
        
        Parameters
        ------------
        coords : mdtraj.Trajectory.xyz object or NumPy ndarray
            Contains the xyz coordinates for all selected atoms
            shape = [n_frames, n_atoms, 3]
        
        Returns
        ------------
        coms : NumPy array of shape = [n_frames, n_residues, 3]
            Contains the centers of mass computed for each residue
            in every frame.
            
        """
            
        com_coords = []
        com_means = []
        _len = coords.shape[0]
        #residues = list(self.rtop.keys())
        #rank = len(residues)
        for i in range(self.nresidues):
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
        #self.chunk_size = chunk_size
        #self.stride = stride
        
    def compute_avg_dist_matrix(self, chunk_size=100, stride=1):
        """Compute the average distance between selected residues
            in a given MD trajectory.
        
        Returns
        ------------
        average distances : NumPy ndarray of shape [n_residues, n_residues]
            
    
        """
        
        start = time.time()
        self.log._logit((2,4))
        
        #residues = list(self.rtop.keys())
        #rank = len(residues)
        avg_coords = None
        iter_ = 0
        #for coords in self.chunks:
        for chunk in md.iterload(self.trajFile, top=self.topFile,
                                 chunk=chunk_size, stride=stride,
                                 atom_indices=self.indices):

            coords = chunk.xyz
            sum_coords = np.sum(coords, axis=0)
            if avg_coords is None:
                avg_coords = sum_coords
            else:
                avg_coords += sum_coords
            iter_ += coords.shape[0]
        avg_coords /= float(iter_)
        
        if self.MEM_OK == True:
            dist_matrix = tst.scipy_dist(avg_coords)
        
            del avg_coords
        
            avg_dist_matrix = np.zeros(shape=(self.nresidues,self.nresidues))
            tst._squeeze(dist_matrix, avg_dist_matrix, self.residues)
        
            del dist_matrix
            
        else:
            self.log._generic("Warning: NATOMS > MEM_ALLOC\nComputing distances in batches" + \
                             " - This may take longer than using SciPy")
            from _batches import gen_batches, batch_distances
            
            avg_dist_matrix = np.zeros(shape=(self.nresidues, self.nresidues))
            
            # Ideal batchsize seems to be 5000
            batchsize = 5000
            
            # Generate batches and compute distances
            batches = gen_batches(self.residues, batchsize)
            batch_distances(self.residues, batches, avg_coords, 
                            avg_dist_matrix)
            
            del avg_coords
            
            avg_dist_matrix = avg_dist_matrix * 10 # Convert to angstroms
        
        self.log._timing(4, round(time.time()-start,3))
        
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

        #rank = len(list(self.rtop.keys()))
        covar = np.zeros(shape=(self.nresidues, self.nresidues))
        iter_ = 0
        w_sum = 0
        for chunk in self.com_data:
            if enable_cuda == True:
               cov = uc.compute_covariances_CUDA(chunk)
            else:
               cov = tst.cov(chunk[0], chunk[1])
            covar += cov
            w_sum += (1*self.chunk_weights[iter_])
            iter_ += 1
        covar /= w_sum
        
        return covar

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
         None - self.corr attribute becomes accessible
         self.corr == correlation matrix : NumPy array of shape [n_residues, n_residues]

        """
        
        start = time.time()
        self.log._logit((2,5))
        
        if enable_cuda == True:
           import utilCUDA as uc

        #rank = len(list(self.rtop.keys()))
        self.corr = np.zeros(shape=(self.nresidues, self.nresidues))
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
            
        self.log._timing(5, round(time.time()-start,3))

    def postprocess(self, scheme='i+1'):
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
        None - correlation matrix is modified in-place

        """
        interval = int(scheme.split('+')[-1])
        rank = self.nresidues
        for i in range(rank-interval):
            ind = [i+j+1 for j in range(interval)]
            for k in ind:
                self.corr[i][k] = 0.0
                self.corr[k][i] = 0.0

    def build_network(self, chunk_size=100, stride=1, align=False, 
                      cutoff=4.5, scheme='i+1', enable_cuda=False):
        """Low-level API that builds the network from the pre-processed
        chunks.
        
        Parameters
        ------------
        chunk_size : int
            Number of frames to process at one time
        stride : int
            Number of frames to skip when processing
        align : bool
            Whether to align the trajectory prior to computing correlation matrix
        cutoff : float 
            Distance cutoff in angstroms for determining in contact residues
        scheme : string
            How to modify the correlation matrix, i+1, i+2...
            or all - no modification
        enable_cuda : bool
            Whether to offload correlation calculation onto an available
            GPU.
            
        Returns
        ------------
        None - distance, correlation and network matrices are saved to file.
        The attributes self.dist_matrix, self.corr and self.network become 
        accessible.
        
        """
        
        start = time.time()
        params = {"Chunk size": chunk_size, "Stride": stride,
                  "Align": align, "Cutoff": cutoff, "Scheme": scheme,
                  "Enable CUDA": enable_cuda}
        self.log._logit((1,7), params=params)
        
        self.dist_matrix = self.compute_avg_dist_matrix(chunk_size=chunk_size,
                                                       stride=stride)
        
        self.process_input(chunk_size=chunk_size, stride=stride, align=align)
        self.compute_correlation_matrix(log=True, enable_cuda=enable_cuda)
        
        self.log._generic("Saving distance and correlation matrices")
        np.savetxt("dist.dat", self.dist_matrix, fmt="%0.5f")
        np.savetxt("corr.dat", self.corr, fmt="%0.5f")
        
        # Modify the distance matrix to include only distances
        # within the cutoff
        self.log._generic("Converting distance matrix to contacts and saving to file")
        self.dist_matrix[self.dist_matrix > cutoff] = 0.0
        self.dist_matrix[self.dist_matrix != 0.0] = 1.0
        np.savetxt("contactmap.dat", self.dist_matrix, fmt="%0.5f")
        
        self.log._generic("Processing scheme and saving network to file")
        if scheme != 'all':
           self.postprocess(scheme=scheme)
        self.network = self.dist_matrix * self.corr
        np.savetxt("network.dat", self.network, fmt="%0.5f")
        
        self.log._timing(7, round(time.time()-start,3))

class DifferenceNetwork(Topology):
    """Builds the network from the input topology and trajectories
    using the difference in persistent contacts between each input. Note 
    that multiple input trajectories are necessary for this calculation to succeed.
    Additionally, all trajectories must have the same number of atoms. In some cases,
    this means that prepocessing is required to ensure that a single topology can be
    employed across every input trajectory.
    
    See Yao, X. Q., Momin, M., & Hamelberg, D. (2019). Journal of chemical information and modeling, 59(7), 3222-3228.
    
    Parameters
    ------------
    topFile : string 
        Path to topology file
    trajFiles : list
        Each item in the list is a string with the path to 
        each trajectory file
    
    """
    
    def __init__(self, top, trajFiles):
        self.top = top
        self.trajFiles = trajFiles
        super(DifferenceNetwork, self).__init__(top)
        self.rtop, self.indices = self.init_top()
        
        self.log = LOG("loggy.log", overwrite=True)
        self.log._startup()
        params = {"Topology": top, "Number of atoms": self.natoms,
                  "Number of residues": self.nresidues}
        for x in range(len(self.trajFiles)):
            t = self.trajFiles[x]
            params["Trajectory {}".format(x+1)] = t
        self.log._logit((0,1), params=params)
        
        self.MEM_OK = True
        if self.natoms > 15999:
            self.MEM_OK = False
         
    def compute_contacts(self, traj, chunk_size=1000, stride=1, 
                        enable_cuda=False, index=0, cutoff=12.0):
        """Computes contacts between all residues in reduced topology.
           Contacts between residues are defined as being within 4.5 angstroms
           of each other. For CUDA versions on systems that do not fit into 
           memory, all atom contacts are computed for a reference frame. This 
           calculation is then used to determine residues within a cutoff for 
           subsquent calculations that can dumped onto the GPU.

        Parameters
        -------------
        traj : string
               Path to trajectory file
        chunk_size : int
               How many frames to process at one time
        stride : int
               Whether to configure the stride argument so that only a 
               percentage of the total trajectory is used
        enable_cuda : bool
               Whether to enable CUDA for computing contacts
        index : int
               Reference frame of trajectory. Only used if enable_cuda == True
               and self.MEM_OK == False
        cutoff : float
               Cutoff in angstroms for reference calculation. Only used if 
               enable_cuda == True and self.MEM_OK == False

        Returns
        -------------
        contact map : NumPy array
               N x N matrix containing contact probabilities between all residues
        
        """
        
        start = time.time()
        self.log._logit((2,6))
        
        # Convert cutoff from angstroms to nanometers
        cutoff = cutoff * 0.1
            
        c = np.zeros(shape=(self.nresidues, self.nresidues))
        tframes = 0
        
        # Case 1: System fits into memory and CUDA is enabled
        if enable_cuda == True and self.MEM_OK == True:
            import utilCUDA as uc
            
            for chunk in md.iterload(traj, top=self.top, chunk=chunk_size,
                                     stride=stride, atom_indices=self.indices):
                coords = chunk.xyz
                tmp_c = uc.contacts_by_frame_CUDA(coords)
                tst._squeeze(tmp_c, c, self.residues, use_min=False)
                tframes += coords.shape[0]
        
        # Case 2: System does or does not fit into memory and we are computing 
        # contacts by frame - Slowest version
        if enable_cuda == False:
            for chunk in md.iterload(traj, top=self.top, chunk=chunk_size,
                                    stride=stride, atom_indices=self.indices):
                coords = chunk.xyz
                for frame in coords:
                    tst.contacts_by_frame(frame, self.residues, c)
                tframes += coords.shape[0]
        
        # Case 3: System does not fit into memory. We compute all distances
        # using a reference frame and batches. Then use a cutoff to determine 
        # which residue pairs will be included in the calculation for the entire trajectory.
        if self.MEM_OK == False and enable_cuda == True:
            from _batches import _gen_batches, batch_distances, _accumulate, gen_nonzero
            from utilCUDA import _reshape
            
            # Load reference frame coordinates
            frame = md.load_frame(traj, index, top=self.top, atom_indices=self.indices)
            coords = frame.xyz[0]
            tmp_c = np.zeros(shape=(self.nresidues, self.nresidues))
            
            # Ideal batch size seems to be 5000
            batchsize = 5000
            
            # Generate batches and compute distances for reference frame
            self.log._generic("Computing all distances for reference frame {}".format(index))
            batches = gen_batches(self.residues, batchsize)
            batch_distances(self.residues, batches, coords, 
                            tmp_c)
            
            # Nonzero elements from reference frame calculation
            w = gen_nonzero(tmp_c, cutoff)
            
            del coords
            del frame
            del tmp_c
            
            self.log._generic("Finished with reference frame, proceeding with full trajectory")
            
            for chunk in md.iterload(traj, top=self.top, chunk=chunk_size,
                                     stride=stride, atom_indices=self.indices):
                coords = _reshape(chunk.xyz)
                _accumulate(w, coords, self.residues, c, use_cuda=True)
                tframes += coords.shape[0]
              
        # Check, in case we've counted contacts twice!
        c[c > tframes] = tframes
        c /= float(tframes)
        
        self.log._timing(6, round(time.time()-start,3))
        
        return c

    def consensus_network(self, states, cutoff):
        """Computes the consensus network from the determined
        contact maps of the input trajectories.
        
        Parameters
        ------------
        states : list
            List of NumPy arrays containing the contact maps
        cutoff : float
            Value at which to consider contacts persistent
            
        Returns
        ------------
        consensus matrix : NumPy NxN array where N == number of 
            conserved residues across the inputs
            
        """
        
        start = time.time()
        self.log._logit((2,8))
        
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
        
        self.log._timing(8, round(time.time()-start,3))
        
        return consensus_matrix

    def diff_network(self, states):
        """Computes the difference network from the determined
        contact maps of input trajectories.
        
        Parameters
        ------------
        states : list
            List containing the contact maps
            
        Returns
        ------------
        difference network : NumPy NxN array where N == number
            of conserved residues across the inputs
            
        """
        
        start = time.time()
        self.log._logit((2,9))
        
        diff = states[0]
        
        for i in range(1, len(states)):
            diff -= states[i]
            
        self.log._timing(9, round(time.time()-start,3))
        
        return diff

    def build_network(self, cutoff1=0.90, chunk_size=100, stride=1,
                     enable_cuda=False, index=0, cutoff2=12.0):
        """Low-level API that builds the network
        
        Parameters
        ------------
        cutoff1 : float 
           Probability cutoff for determining persistent contacts
        chunk_size : int
            Number of frames to process at one time
        stride : int
            Configure stride to only use a percentage of the total
            trajectory
        enable_cuda : bool
            Whether to use CUDA version for computing contacts
        index : int
            Frame number for reference calculation, only used when enable_cuda==True
            and self.MEM_OK==False
        cutoff2 : float
            Distance in angstroms for computing distances in reference frame,
            only used when enable_cuda==True and self.MEM_OK==False
            
        Returns
        ------------
        None - consensus and difference matrices are saved to file. The
            attributes self.consensus_matrix and self.difference_matrix
            become accessible.
        
        """
        
        start = time.time()
        params = {"Cutoff": cutoff1, "Chunk size": chunk_size,
                  "stride": stride, 
                  "Number of states": len(self.trajFiles)}
        self.log._logit((1,7), params=params)
        
        states = []
        
        for i in range(len(self.trajFiles)):
            
            current_traj = self.trajFiles[i]
            
            state = self.compute_contacts(current_traj, chunk_size, 
                                          stride=stride,
                                          enable_cuda=enable_cuda,
                                          index=index, cutoff=cutoff2)
            states.append(state)
            np.savetxt("state{}.dat".format(i+1), state, fmt="%0.5f")
       
        self.consensus_matrix = self.consensus_network(states, cutoff=cutoff1)
        self.difference_matrix = self.diff_network(states)
        
        self.log._generic("Saving consensus and difference matrices to file")
        np.savetxt("consensus.dat", self.consensus_matrix, fmt="%0.0f")
        np.savetxt("difference.dat", self.difference_matrix, fmt="%0.5f")
        
        self.log._timing(7, round(time.time()-start, 3))
                
    def get_communities(self, commFile, offset):
        """Function for loading communities from a txt file.
        
        Parameters
        ------------
        commFile : string
            Path to communities file
        offset : int
            Where the residue numbering starts - in most cases
            this will be zero.
        
        Returns
        ------------
        communities - Python dict
            Dict is structured as keys == community ID (index-based) 
            and values == residue IDs that belong to that community
            (also index-based).
        
        """
        
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
        """Function for accumulating the difference in contact
        probabilities into communities.
        
        commFile : string
            Path to communities text file
        diff : NumPy array
            Difference network containing the change in 
            contact probabilities
        offset : int
            Where the residue numbering starts - in most cases
            this will be zero.
            
        Returns
        -----------
        contacts : Python dict
            Dict is structured as keys == tuple(communityi, communityj)
            and values == total difference in contact probability
            
        """
        
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
        """Writes the results of self.deltaP to a text file.
        
        Parameters
        -------------
        deltaP : Python dict
            contacts determined from self.deltaP
        """
        
        f = open(oname, "w")
        for pair in deltaP:
            f.write("{} {} {}\n".format(pair[0], pair[1], deltaP[pair]))
        f.close()

