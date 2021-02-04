# MDNetworkTools

MDNetworkTools is an all-in-one python package that builds networks directly from molecular dynamics trajectories. Additionally, the software
provides functionality for network analysis (community detection and suboptimal paths calculations), as well as, visualization. 
At its core, MDNetworkTools utilizes multiple Python modules. Trajectory and topology processing is accomplished with MDTraj, while the network generation
algorithms employ jit-compiled methods via the Numba package. At present, only network generation methods benefit from the speed-up of Numba, although future releases
may extend this to the network analysis algorithms. Please refer to Required Packages for the complete list of necessary software.

# Required Packages
In addition to numpy and scipy, the following packages will need to be installed prior to running MDNetworkTools:
1) MDTraj - https://mdtraj.org/1.9.4/index.html
2) Numba - https://numba.pydata.org/
3) Networkx - https://networkx.org/documentation/stable/

# Installation
To install this package, first clone this repository:

git clone https://github.com/tdodd3/mdnetworktools

Then:

cd mdnetworktools/

python setup.py install

# Basic Usage
While MDNetworkTools allows for advanced Python scripting, most users will be able to get away with using
the high-level API executable found in the /bin directory. Upon successful installation:

cd bin/
chmod +x mdnetwork

Then edit the $PATH variable in your .bashrc file. As an example:

export PATH=$PATH:/home/tdodd/mdnetworktools/bin/

Now, you can call mdnetwork executable along with a configuration file from the command line. 

mdnetwork example.cfg

Please see the /examples directory for examples of .cfg files and advanced scripts.

# Using CUDA
MDNetworkTools has some features which can take advantage of a GPU. Currently, there are two ways to 
enable the CUDA versions of these methods:

1) Install cudatools via the Conda installer (see numba docs).
2) If CUDA is already installed on your system, then you can source a bash file prior to running MDNetworkTools.
Refer to env_example.sh in this directory for examples with 2 different setups (python 2.7, numba 0.43.1 and CUDA-8.0)
and (python 3.7, numba 0.52.1 and CUDA-9.0).
