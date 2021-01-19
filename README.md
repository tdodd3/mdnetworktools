# MDNetworkTools

MDNetworkTools is an all-in-one python package that builds networks directly from molecular dynamics trajectories. Additionally, the software
provides functionality for network analysis and visualization. 

# Required Packages
In addition to numpy and scipy, the following packages will need to be installed prior to running MDNetworkTools:
1) mdtraj - https://mdtraj.org/1.9.4/index.html
2) numba - https://numba.pydata.org/
3) networkx - https://networkx.org/documentation/stable/

MDNetworkTools has some features which can take advantage of a GPU. To enable CUDA versions of 
some fuctions, either cudatools must be installed through conda (see numba docs), or you must source a bash file
prior to running your script. An example bash file can be found in this directory (numbaVAR_example.sh).

# Installation
To install this package, first clone this repository:

git clone https://github.com/tdodd3/mdnetworktools

Then:

python setup.py install
