# MDNetworkTools

MDNetworkTools is an all-in-one python package that builds networks directly from molecular dynamics trajectories. Additionally, the software
provides functionality for network analysis (community detection and subtoptimal paths calculations) and visualization. 
At its core, MDNetworkTools utilizes multiple Python modules. Trajectory and topology processing is accomplished with MDTraj, while the network generation
algorithms employ jit-compiled methods via the Numba package. At present, only network generation methods benefit from the speed-up of jit, although future releases
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

python setup.py install

# Using CUDA
MDNetworkTools has some features which can take advantage of a GPU. Currently, there are two ways to 
enable the CUDA versions of these methods:

1) Install cudatools via the Conda installer (see numba docs).
2) If CUDA is already installed on your system, then you can source a bash file prior to running MDNetworkTools.
Refer to numbaVAR_example.sh in this directory for examples with 2 different setups (python 2.7, numba 0.43.1 and CUDA-8.0)
and (python 3.7, numba 0.52.1 and CUDA-9.0).
