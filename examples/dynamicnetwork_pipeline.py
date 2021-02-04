"""
	Pipeline example using the DynamicNetwork object to build a network from a MD trajectory. 
	The DynamicNetwork object builds the network using cross-correlation and residues 
	within 4.5 angstroms of each other. Draw the results in the BILD format which is recognized
	by Chimera. Run the Girvan-Newman algorithm to find a user-specified number of communities. 
	Generate a python script to be read by Chimera to visualize the results of Girvan-Newman. Finally,
	compute the first 1000 suboptimal paths using the SOAN method and write the paths in the BILD format. 
	
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
		**** chunk_size is at least 1000 frames and stride is at least 2. This will decrease the memory   ****
		**** overhead and greatly speed up the generation of the network while also maintaining accuracy.****
		
	Parameters for DrawObject
	
		top : string		Path to topology file
		ref : string		Path to reference PDB (the network is mapped to this structure)
		network : array		Adjacency matrix to be mapped to reference structure, default is False
		
	Parameters for DrawObject.write_chimera_session
		
		ncomms : int		The subdivision to use for coloring communities
		
	Parameters for DrawObject.draw_paths
	
		ifile : string		Path to suboptimal paths file - if this was generated with SOAN it will be 'paths.txt' (default)
		pathcolor : string	Coloring of the paths in the BILD file
		smoothness : float	For smoothing the paths using interpolation - the default 0.01 works well
		radius : float		Radius of the tubes for path depiction - the default of 0.1 works well
		output : string		Name of output file, default is 'paths.bild'
		
	Parameters for GirvanNewman object
	
		network : array		Can be a DynamicNetwork.network object or an array loaded from a text file
		
	Parameters for GirvanNewman.run API
	
		weight : var		Whether to use the weights found in the network - in the case of an unweighted network 
							this should be set to None. Default is 'weight'.
		ncomms : int		Number of communities to find before terminating, default is 25.
		
	Parameters for SOAN
	
		A : array			Network, can be a DynamicNetwork.network object or one loaded from a text file
		s : int				Source residue - this needs to be serial-based as it is converted to an index internally
		t : int				Target residue - this needs to be serial-based as it is converted to an index internally

	Parameters for SOAN.run API
	
		level : int			How many neighbors away from the optimal path to expand to, default is 1
		numpaths : int		The number of suboptimal paths to find at the given expansion level
		
		**** See reference ... for a detailed description of the SOAN method, ****
		**** as well as, discussion on the level and numpaths parameters.     ****
		
		
"""

from mdnetworktools.builds import DynamicNetwork
from mdnetworktools.algorithms import GirvanNewman, SOAN
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

# Run Girvan-Newman on the computed network
gn = GirvanNewman(net.network)
gn.run(weight='weight', ncomms=10)

# Generate python script to be loaded by Chimera to visualize communities
drawObj.write_chimera_session(ncomms=10)

# Compute 1000 suboptimal paths between residues 403 and 1820 using the SOAN method
subopt = SOAN(net.network, s=403, t=1820)
subopt.run(level=2, numpaths=1000)

# Draw the paths in the BILD format
drawObj.draw_paths(pathcolor='red', smoothness=0.01, radius=0.1, output='paths.bild')
  					