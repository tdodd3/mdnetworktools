"""
	Simple example using the DynamicNetwork object to build a network from a MD trajectory. 
	The DynamicNetwork object builds the network using cross-correlation and residues 
	within 4.5 angstroms of each other. Finally, draw the results in the BILD format which is recognized
	by Chimera.
	
	Parameters for DynamicNetwork object 
	
		top : string 		Path to topology file
		traj : string		Path to trajectory file
		ref : string		Path to reference PDB (only specify if the trajectory needs to be aligned), default is None
	
	There are several parameters for the DynamicNetwork.build_network API:
		
		chunk_size : int	How many frames to process at one time, default is 100
		stride : int		How many frames to skip when processing, default is 1
		align : int 		Whether to align the trajectory to a reference (one must be provided), default is False
		cutoff : float 		The distance (in angstroms) at which residues are considered in-contact, default is 4.5 angstroms
		scheme : string		Modification of the correlation matrix (i+1 == set cc for neighboring residues to zero), default is 'i+1'
		enable_cuda : bool	Whether to offload correlation calculation onto an available GPU, default is False
		
		**** Please note that for trajectories containing more than 10,000 frames it is recommended that ****
		**** chunk_size is at least 1000 frames and stride is at least 2. This will improve the memory   ****
		**** overhead and greatly speed up the generation of the network while also maintaining accuracy.****
		
	Parameters for DrawObject
	
		top : string		Path to topology file
		ref : string		Path to reference PDB (the network is mapped to this structure)
		network : array		Adjacency matrix to be mapped to reference structure, default is False
		
"""

from mdnetworktools.builds import DynamicNetwork
from mdnetworktools.visualization import DrawObject

top = "example.prmtop" # path to topology
traj = "example.dcd" # path to trajectory
ref = "example.pdb" # for visualization

# Build the network 
net = DynamicNetwork(top, traj)
net.build_network(chunk_size=1000, stride=2, align=False, 
					cutoff=4.5, scheme='i+1', enable_cuda=False)

# Draw the network
drawObj = DrawObject(top, ref, network=net.network)
drawObj.draw_network() 					