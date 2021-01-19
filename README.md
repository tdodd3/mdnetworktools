# MDNetworkTools

MDNetworkTools is an all-in-one python package that builds networks directly from molecular dynamics trajectories. Additionally, the software
provides functionality for network analysis (Community detection and Subtoptimal paths calculations) and visualization. 
At its core, MDNetworkTools utilizes existing python modules for trajectory processing (mdtraj), as well as, network generation (by employing
jit-compiled methods through numba). At present, only network generation methods benefit from the speed-up of jit, although future releases
may extend this to the network analysis algorithms. Please refer to Required Packages for the complete list of necessary software.

# Required Packages
In addition to numpy and scipy, the following packages will need to be installed prior to running MDNetworkTools:
1) mdtraj - https://mdtraj.org/1.9.4/index.html
2) numba - https://numba.pydata.org/
3) networkx - https://networkx.org/documentation/stable/

# Installation
To install this package, first clone this repository:

git clone https://github.com/tdodd3/mdnetworktools

Then:

python setup.py install

# Using CUDA
MDNetworkTools has some features which can take advantage of a GPU. Currently, there are two ways to 
enable the CUDA versions of these methods:

1) Install cudatools via Conda installer (see numba docs).
2) If you have aleady have CUDA installed on your system then you can source a bash file to point to
the $CUDA_HOME environment variable prior to running MDNetworkTools. An example bash file is given in this directory.
