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
import networkx as nx
import networkx.algorithms.community as nac
from operator import itemgetter
from itertools import count
from heapq import heappush, heappop
from _logger import LOG
import time

class GirvanNewman(object):
	"""Executes the Girvan-Newman algorithm to subdivide
	the input network into a specified number of communities.
	
	Parameters
	------------
	network : NumPy ndarray
		This can either be a mdnetworktools object (DynamicNetwork.network
		DifferenceNetwork.network) or a network that has been loaded
		into memory using numpy.genfromtxt.
		
	"""
	
	def __init__(self, network):
		self.network = network
		self.G = nx.from_numpy_matrix(network)
		self.log = LOG("loggy.log", overwrite=False)
		
		self.log._startup()
	#### Helper functions ####
	
	# Betweenness
	def gnStep(self, weight='weight'):
		init_ncomp = nx.number_connected_components(self.G)
		ncomp = init_ncomp
		while ncomp <= init_ncomp:
			bw = nx.edge_betweenness_centrality(self.G, weight='weight')
			max_ = max(bw.values())
			# Find edge with highest centrality and remove
			for k, v in bw.iteritems():
				if float(v) == max_:
					self.G.remove_edge(k[0], k[1])
			ncomp = nx.number_connected_components(self.G)
			
	# Modularity
	def compute_modularity(self, clist):
		comps = []
		for c in clist:
			s = {x for x in c}
			comps.append(s)
		return nac.modularity(self.G, comps)
	
	# Preprocessing 
	def _remove(self, len_comps=4):
		llfile = open("removed_components.txt", "w")
		icomps = sorted(nx.connected_components(self.G),
				key=len, reverse=True)
		len_icomps = len(icomps)
		llfile.write("Number of components before connectivity check: " + \
				"{}\n".format(len_icomps))
		for ic in icomps:
			if len(ic) < len_comps:
				for lc in ic:
					llfile.write("{}\n".format(lc))
				self.G.remove_nodes_from(ic)
		ncomps = sorted(nx.connected_components(self.G),
				key=len, reverse=True)
		len_ncomps = len(ncomps)
		if len_icomps == len_ncomps:
			llfile.write("No nodes removed\n")
		else:
			llfile.write("Number of connected components: " + \
					"{}\n".format(len_ncomps))
	
	#### Main execution ####
	
	def run(self, weight='weight', ncomms=25):
		"""Run the Girvan-Newman algorithm on the input network.
		
		Parameters
		------------
		weight : string or None
			For dynamic networks (mdnetworktools.DynamicNetwork)
			weight should be set to 'weight'. For difference 
			networks (mdnetworktools.DifferenceNetwork) weight
			should be set None.
		ncomms : int
			Number of communities at which to stop the algorithm
			
		Returns
		-----------
		None - The community information is saved to a file at every 
			subdivision.
			
		"""
		
		start = time.time()
		params = {"Weight": weight, "Number of communities": ncomms}
		self.log._logit((0,2), params=params)
		
		self._remove() # Remove loosely-connected components from the graph
		t = open("communities.dat", "w")
		bestQ = 0.0
		diff = 0.0
		n_communities = 0
		while n_communities < ncomms:
			self.gnStep(weight=weight)
			ccomps = sorted(nx.connected_components(self.G),
					key=len, reverse=True)
			Q = self.compute_modularity(ccomps)
			diff = abs(bestQ - Q)
			n_communities = len(ccomps)
			if Q > bestQ:
				bestQ = Q
				bestcomps = ccomps
				self.log._generic("Partition {}: Modularity: {} " + \
						  "Difference from previous {}".format(n_communities,
										      Q, diff))
			if n_communities > 1:
				t.write("Partition {}: ".format(n_communities))
				for c in ccomps:
					for node in c:
						t.write("{} ".format(node))
					t.write("\n")
			if self.G.number_of_edges() == 0:
				break
				
		self.log._timing(2, round(time.time()-start))
		
		t.close()

class PathBuffer(object):
	"""
	Heap-like priority queue modified for paths determined with 
	Yen's algorithm. This data structure is analgous to the one
	used in NetworkX.
	"""
	def __init__(self):
		self.paths = set()
		self.sortedpaths = list()
		self.counter = count()
	
	def __len__(self):
		return len(self.sortedpaths)
	
	def push(self, cost, path):
		hashable_path = tuple(path)
		if hashable_path not in self.paths:
			heappush(self.sortedpaths, (cost, next(self.counter), path))
			self.paths.add(hashable_path)
		
	def pop(self):
		(cost, num, path) = heappop(self.sortedpaths)
		hashable_path = tuple(path)
		self.paths.remove(hashable_path)
		return path

class SOAN(PathBuffer):
	"""Computes suboptimal paths between a source and target 
	node using Subsets of Adjacent Nodes (SOAN) method. 
	
	See reference
	
	Parameters
	-------------
	A : NumPy ndarray
		Network matrix - can be a mdnetworktools object or
		loaded into memory with numpy.genfromtxt
	s : int
		Source node (serial-based)
	t : int
		Target node (serial-based)
		
	"""
	
	def __init__(self, A, s, t):
		self.A = A
		self.G = nx.from_numpy_matrix(self.A)
		self.s = s - 1 # convert to index
		self.t = t - 1 # convert to index
		self.log = LOG("loggy.log", overwrite=False)
		
		self.log._startup()

	def dijkstra(self, A, s, t, ignore_nodes=None, ignore_edges=None):
		"""
		This is a variant of Dijkstra's algorithm that uses a 
		bidirectional search to find the shortest path. The code
		structure is similar to that of NetworkX with the exception
		that it has been modified to work explicitly on NumPy arrays.
		Additionally, the parameter ignore_edges/nodes allows one to 
		employ this algorithm with Yen's algorithm.
		
		Parameters
		------------
		A : NumPy ndarray
			Network matrix
		s : int
			Source node (index-based)
		t : int 
			Target node (index-based)
		ignore_nodes : set or None
			Nodes to ignore when computing the shortest path
		ignore_edges : set of tuples == (i, j) where i,j make the edge or None
			Edges to ignore when computing the shortest path
			
		Returns
		-----------
		shortest path : list of ints
			Indices of the nodes that comprise the shortest path found
			In the event that No path is found, raises KeyError - don't
			worry this is handled elsewhere.
				
		"""
		if s == t:
			return (0, [s])

		push = heappush
		pop = heappop
		dists = [{}, {}]
		paths = [{s: [s]}, {t: [t]}]
		fringe = [[], []]
		seen = [{s: 0}, {t: 0}]
		c = count()
		push(fringe[0], (0, next(c), s))
		push(fringe[1], (0, next(c), t))
		finalpath = []
		dir = 1
		while fringe[0] and fringe[1]:
			# choose direction, 0 is forward
			dir = 1 - dir
			(d, _, current_node) = pop(fringe[dir])
			neighbors = np.where(A[:,current_node] != 0.0)[0]
			if ignore_nodes:
				def filter_nodes(neigh):
					for n in neigh:
						if n not in ignore_nodes:
							yield n
				neighbors = filter_nodes(neighbors)
				neighbors = list(neighbors)
			if ignore_edges:
				def filter_edges(neigh):
					for n in neigh:
						if (current_node, n) not in ignore_edges \
							and (n, current_node) not in ignore_edges:
							yield n
				neighbors = filter_edges(neighbors)
				neighbors = list(neighbors)
			if current_node in dists[dir]:
				continue
			dists[dir][current_node] = d
			if current_node in dists[1-dir]:
				return (finaldist, finalpath) # We are done
			if len(neighbors) == 0:
				raise KeyError
			for n in neighbors:
				cost = A[current_node][n]
				cn_dist = dists[dir][current_node] + cost
				if n in dists[dir]:
					if cn_dist < dists[dir][n]:
						raise ValueError("Negative weights?")
				elif n not in seen[dir] or cn_dist < seen[dir][n]:
					seen[dir][n] = cn_dist
					push(fringe[dir], (cn_dist, next(c), n))
					paths[dir][n] = paths[dir][current_node] + [n]
				if n in seen[0] and n in seen[1]:
					totaldist = seen[0][n] + seen[1][n]
					if finalpath == [] or finaldist > totaldist:
						finaldist = totaldist
						revpath = paths[1][n][:]
						revpath.reverse()
						finalpath = paths[0][n] + revpath[1:]
		if finalpath == []:
			raise KeyError

	#### Helper Functions ####		
	
	# Optimal path
	def find_opt_path(self):
		opt_path_len, opt_path = self.dijkstra(self.A, self.s, self.t)
		self.opt_path = opt_path
		self.opt_path_len = opt_path_len
	
	# Neighbors by node
	def _neighbors(self, node):
		neighbors = np.where(self.A[:,node] != 0.0)[0]
		neighbors = [x for x in neighbors if x != node]
		return neighbors
	
	# Neighbors of all given nodes
	def get_all_neighbors(self, path):
		all_neighbors = []
		for node in path:
			all_neighbors.append(self._neighbors(node))
		all_neighbors = np.hstack(all_neighbors)
		return list(set(all_neighbors))

	# Level expansion
	def get_level(self):
		if self.level == 1:
			return self.get_all_neighbors(self.opt_path)
		else:
			master = []
			master.append(self.get_all_neighbors(self.opt_path))
			count = 0
			while count < self.level:
				master.append(self.get_all_neighbors(master[count]))
				count += 1
			master = np.hstack(master)
			return list(set(master))

	# Subgraph creation at user-specified level
	def create_subgraph(self, level=1):
		self.level = level
		neighborlist = self.get_level()
		paths = [self.opt_path]
		for node in neighborlist:
			node_path = nx.all_shortest_paths(self.G, source=self.s,
							target=node,
							weight='weight')
			target_path = nx.all_shortest_paths(self.G, source=node,
							target=self.t,
							weight='weight')
			for p in node_path:
				for j in target_path:
					try:
						if j[1] not in p:
							path = p[:-1] + j
							if path not in paths:
								paths.append(path)
					except IndexError:
						pass
		nodelist = np.hstack(paths)
		nodelist = list(set(nodelist))
		self.S = self.G.subgraph(nodelist)
		unmapped_paths = paths
		mapping = {list(self.S.nodes())[n]: n for n in range(len(nodelist))}
		reversemapping = {n: list(self.S.nodes())[n] for n in \
					range(len(nodelist))}
		SadjM = nx.to_numpy_array(self.S, dtype=np.float64)
		rtarget = mapping[self.t]
		return SadjM, mapping, reversemapping, unmapped_paths, rtarget

	# Postprocessing
	def map_paths_to_graph(self, paths, mapping):
		mapped_paths = []
		for p in paths:
			mapped_paths.append([mapping[x] for x in p])
		return mapped_paths

	# Distance
	def _dist(self, A, path):
		return np.sum([A[path[x]][path[x+1]] for x in range(len(path)-1)])
	
	# Yen's algorithm
	def find_paths_from_graph(self, S, k, t, somepath):
		"""
		This is Yen's algorithm for finding k-shortest paths.
		It uses Dijkstra's algorithm to find the shortest path
		between a spur and the target. Additionally, we ignore
		nodes and edges from already discovered paths.
		
		Parameters
		------------
		S : NumPy ndarray
			Subgraph or subset of nodes to search for 
			suboptimal paths.
		k : int
			Number of shortest paths to find in given
			subgraph
		t : int
			target node - mapped to the subgraph
		somepath : list
			Already discovered path - used to populate
			an empty Queue object. This is usually the optimal
			path.
			
		Returns
		-----------
		Suboptimal paths : list of lists
			Each list contains the nodes that comprise one of the
			shortest paths found with Yen's algorithm.
			
		"""
		
		found_paths = [somepath]
		pathlist = [somepath]
		prev_path = somepath
		Qpath = PathBuffer()
		EXIT = False
		while len(found_paths) < k:
			ignore_nodes = set()
			ignore_edges = set()
			for i in range(1, len(prev_path)):
				root = prev_path[:i]
				root_len = self._dist(S, root)
				for p in pathlist:
					if p[:i] == root:
						ignore_edges.add((p[i-1], p[i]))
				try:
					length, spur = self.dijkstra(S, root[-1], t,
								ignore_nodes=ignore_nodes,
								ignore_edges=ignore_edges)
					path = root[:-1] + spur
					path_len = root_len + length
					Qpath.push(path_len, path)
				except KeyError:
					pass
				except TypeError: # We've run out of nodes
					EXIT = True
					break
				ignore_nodes.add(root[-1])
			if EXIT == True:
				break
			if Qpath:
				new_path = Qpath.pop()
				pathlist.append(new_path)
				if new_path not in found_paths:
					found_paths.append(new_path)
				prev_path = new_path
			else:
				break
		return found_paths

	# Saves the results to file
	def output_paths(self, paths, pathstowrite):
		unsorted = [p + [self._dist(self.A, p)] for p in paths]
		sorted_paths = sorted(unsorted, key=itemgetter(-1))
		t = open("paths.txt", "w")
		c = 0
		for p in sorted_paths:
			s = " ".join([str(x) for x in p[:-1]])
			t.write(s+" {}\n".format(p[-1]))
			c += 1
			if c > pathstowrite:
				break
		t.close()

		
	def run(self, level=1, numpaths=1000):
		"""Low-level API that runs the SOAN method.
		
		Parameters
		-------------
		level : int
			The level at which to create the subgraph - 
			how many neighbors away from the optimal path
			to expand to.
		numpaths : int
			The total number of shortest paths to find in the 
			subgraph
			
		Returns
		------------
		None - All results are written to file.
		
		"""
		
		params = {"Level": level, "Number of Paths": numpaths}
		self.log._logit((0,3), params=params)
		
		self.log._generic("Searching for optimal path between {} and {}".format(self.s+1, self.t+1))
		s1 = time.time()
		self.find_opt_path()
		self.log._timing(10, round(time.time()-s1, 3))
		
		s2 = time.time()
		self.log._generic("Creating subgraph at level {} of the optimal path".format(level))
		SadjM, mapping, reversemapping, \
		unmapped_paths, mapped_t = self.create_subgraph(level)
		self.log._timing(11, round(time.time()-s2, 3))
		
		self.log._generic("Searching {} suboptimal paths in subgraph".format(numpaths))
		s3 = time.time()
		paths = self.find_paths_from_graph(SadjM, numpaths, mapped_t, 
						[mapping[x] for x in unmapped_paths[0]])
		self.log._timing(12, round(time.time()-s3, 3))
		
		mapped_paths = []
		for p in paths:
			mapped_paths.append([reversemapping[x] for x in p])
		self.output_paths(mapped_paths, numpaths)
		
		
	
