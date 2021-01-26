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
"""
	This is a class of functions for visualizing the results of
	mdnetworktools analyses (i.e. network generation, community detection 
	or suboptimal paths calculations). Results are transformed into the BILD
	format which is recognized by the Chimera visualization software
	https://www.cgl.ucsf.edu/chimera/. For community detection, a python
	script is generated which can be loaded into Chimera for visualization
	purposes. These functions have only been tested on PDBs in which residues have been 
	renumbered sequentially (1,2,3,...,N). However, it is possible that some of
	these methods will still work with general PDBs downloaded from RCSB.
"""

import numpy as np
from scipy import interpolate
import mdtraj as md
from builds import Topology

class DrawObject(Topology):
    def __init__(self, top, ref, network=None):
        super(DrawObject, self).__init__(top)
        self.rtop, self.indices = self.init_top()
        self.network = network
        self.ref = md.load_pdb(ref, atom_indices=self.indices)
	self.refName = ref    

    def com_(self, coords, resid):
        indices = list(self.rtop[resid].keys())
        weights = list(self.rtop[resid].values())
        M = sum(weights)
        coorX = np.sum([coords[indices][:,0]*weights])/M
        coorY = np.sum([coords[indices][:,1]*weights])/M
        coorZ = np.sum([coords[indices][:,2]*weights])/M
        
        return np.asarray([coorX, coorY, coorZ])
    
    def draw_network(self):
        t = open("network.bild", "w")
        coords = self.ref.xyz[0]*10
        # First we need to generate coordinates for the 
        # mapping of each node
        com_coords = []
        for resid in self.rtop.keys():
            com = self.com_(coords, resid)
            com_coords.append(com)
        
        # Now to map the nonzero edges to the nodes
        for res in range(len(com_coords)):
            b = com_coords[res]
            nonzero = np.where(self.network[:,res] != 0.0)[0]
            for a in nonzero:
                c = com_coords[a]
                t.write(".color red\n")
                clt = ".cylinder {} {} {} {} {} {} {}\n".format(round(b[0],3),
					round(b[1],3), round(b[2],3), round(c[0],3), round(c[1],3),
					round(c[2],3), 0.1)
                t.write(clt)
        t.close()

    def import_paths(self, name):
        paths = []
        with open(name) as f:
             for line in f:
                 lines = line.split()
                 paths.append([int(x) for x in lines[:-1]])
        spaths = np.hstack(paths)
        nodes = np.unique(spaths)
        return paths, nodes

    def intrpl_(self, path, coords, smoothness):
        x_vals = []
        y_vals = []
        z_vals = []
        for res in path:
            coor = coords[res]
            x_vals.append(coor[0])
            y_vals.append(coor[1])
            z_vals.append(coor[2])
        degree = len(x_vals) - 1
        if degree > 3:
           degree = 3
        tck, _ = interpolate.splprep([x_vals, y_vals, z_vals], s=0, k=degree)
        unew = np.arange(0, 1.01, smoothness)
        out = interpolate.splev(unew, tck)
        return out
        
    def draw_paths(self, ifile="paths.txt", pathcolor="red", smoothness=0.01, 
                   radius=0.1, output="paths.bild"):
        t = open(output, "w")
        coords = self.ref.xyz[0]*10
        com_coords = []
        for resid in self.rtop.keys():
            com = self.com_(coords, resid)
            com_coords.append(com)
        paths, _ = self.import_paths(ifile)
        for p in paths:
            ipath = self.intrpl_(p, coords, smoothness)
            for c in range(len(ipath[0]) - 1):
                x1 = ipath[0][c]
                y1 = ipath[1][c]
                z1 = ipath[2][c]
                x2 = ipath[0][c+1]
                y2 = ipath[1][c+1]
                z2 = ipath[2][c+1]
                t.write(".color "+pathcolor+"\n")
                lt = ".cylinder {} {} {} {} {} {} {}\n".format(round(x1,3),
                       round(y1,3), round(z1,3), round(x2,3), round(y2,3),
                       round(z2,3), radius)
                t.write(lt)
        t.close()

    def writecommunities(self, commFile, ncomms):

        comms = {}
        fromFile = None
        FILE = open(commFile, "r")
        f = FILE.readlines()
        count = 0
        for line in f:
                if "Partition" in line:
                        lines = line.split()
                        if int(lines[-1]) == ncomms:
                                fromFile = f[count+1:count+ncomms+1]
                                break
                count += 1

        t = open("communities{}.txt".format(ncomms), "w")
        for l in fromFile:
                t.write(l)
        t.close()
        FILE.close()

    def write_chimera_session(self, ncomms, commFile="communities.dat"):

        self.writecommunities(commFile, ncomms)
        t = open("communities{}.py".format(ncomms), "w")

        t.write("import os\nimport numpy as np\n" + \
                "from chimera import runCommand as rc\n" + \
                "from chimera import replyObj\n")

        t.write("\n# Def function for reading in community data\n")

        func = "def get_communities(fname, offset=1):\n\ts={}\n\tcount=0\n" + \
                "\twith open(fname) as f:\n\t\tfor line in f:\n\t\t\t" + \
                "lines = line.split()\n\t\t\tresids = [str(int(r)+offset) for r in lines]\n\t\t\t" + \
                "s[count] = resids\n\t\t\tcount += 1\n\treturn s\n"

        color_dict = "\ncolors = {0:'cornflower blue', 1:'coral', 2:'dark magenta',\n\t\t" + \
                        "3:'dark cyan', 4:'cyan', 5:'forest green', 6:'navy blue',\n\t\t" + \
                        "7:'magenta', 8:'purple', 9:'deep pink', 10:'salmon',\n\t\t" + \
                        "11:'blue', 12:'green', 13:'gold', 14:'rosy brown',\n\t\t" + \
                        "15:'khaki', 16:'red', 17:'pink', 18:'dim gray',\n\t\t" + \
                        "19:'sienna', 20:'dark red', 21:'sea green', 22:'tan',\n\t\t" + \
                        "23:'orchid', 24:'violet red'}\n"

        txtfile = "communities{}.txt".format(ncomms)
        exe = "for c in comms:\n\treplyobj.status('Coloring residues in community ' + str(c))\n\t" + \
		"resids = ','.join(comms[c])\n\trc('color ' + colors[c] + ' :' + resids)"
	t.write(func)
        t.write(color_dict)
        t.write("comms = get_communities('{}')\n".format(txtfile))
        t.write("rc('open ' + '{}')\n".format(self.refName))
        t.write(exe)

        t.close()
