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

"""High-level API that configures options from a .cfg file and runs 
the appropriate mdnetworktools method with the specified options.
"""

from configparser import ConfigParser
import numpy as np
from builds import DynamicNetwork, DifferenceNetwork
from algorithms import GirvanNewman, SOAN
#from mdnetworktools.visualization import DrawObject

class Configuration(object):
	"""Class for running mdnetworktools from
	config files
	"""

	def __init__(self, config_file):
		self.cfg = ConfigParser()
		self.cfg.read(config_file)

		# Config general options
		self.GENERAL = self.cfg["GENERAL"]

		self.top = str(self.GENERAL["topology"])
		self.load_network = self._bool(self.GENERAL["load_network"])
		self.network_name = str(self.GENERAL["network_name"])
		self.reference = str(self.GENERAL["reference"])
		self.visualize = self._bool(self.GENERAL["visualize"])

		

		tmp_trajs = str(self.GENERAL["trajectory"])
		if ',' in tmp_trajs:
			self.trajs = tmp_trajs.split(',')
		else:
			self.trajs = tmp_trajs

		if self.load_network:
			self.network = np.genfromtxt(self.network_name)
		else:
			self.network = None

		self.OPTIONS = {"NETWORK": {}, "COMMUNITIES": {}, 
				"SUBOPTIMALPATHS": {},
				"DELTAP": {}}		

		# Determine what we are doing...
		self.queue = []
		for option in self.cfg.keys():
			if option == "DEFAULT" or option == "GENERAL":
				continue
			else:
				tmp = self.cfg[option]
				self.queue.append(option)
				for p in tmp:
					self.OPTIONS[option][p] = tmp[p]

	def _bool(self, option):
		if option == 'True':
			return True
		else:
			return False

	def _find_true(self, alist):
		s = [i for i in range(len(alist)) if alist[i] == True]
		return s

	def config_and_run_network(self):
		info = self.OPTIONS["NETWORK"]
		network_type = info["type"]
		# chunk, stride and enable_cuda are universal
		chunk_size = int(info["chunk_size"])
		stride = int(info["stride"])
		enable_cuda = self._bool(info["enable_cuda"])
		
		# Dynamic network
		if network_type == 'Dynamic':
			align = self._bool(info["align"])
			cutoff = float(info["cutoff"])
			scheme = str(info["scheme"])

			net = DynamicNetwork(self.top, self.trajs)
			net.build_network(chunk_size=chunk_size,
						stride=stride, align=align,
						cutoff=cutoff, scheme=scheme,
						enable_cuda=enable_cuda)

			self.network = net.network

		# Difference Network
		elif network_type == 'Difference':
			cutoff1 = float(info["contacts_cutoff"])	
			index = int(info["reference_frame"])
			cutoff2 = float(info["distance_cutoff"])

			net = DifferenceNetwork(self.top, self.trajs)

			net.build_network(cutoff1=cutoff1,
						chunk_size=chunk_size,
						stride=stride,
						enable_cuda=enable_cuda,
						index=index, cutoff2=cutoff2)

			self.network = net.consensus_matrix
			#self.diff = net.difference_matrix

		else:
			raise KeyError("Network type not found - Check .cfg file")	

	def config_and_run_analysis(self, _type):
		info = self.OPTIONS[_type]
		if _type == 'COMMUNITIES':
			weight = info["weight"]
			if weight == 'None':
				weight = None
			ncomms = int(info["number_of_communities"])
			
			gn = GirvanNewman(self.network)
			gn.run(weight=weight, ncomms=ncomms)

		elif _type == 'SUBOPTIMALPATHS':
			source = int(info["source"])
			target = int(info["target"])
			level = int(info["level"])
			numpaths = int(info["number_of_paths"])

			soan = SOAN(self.network, source, target)
			soan.run(level=level, numpaths=numpaths)

		elif _type == 'DELTAP':
			community_file = info["community_file"]
			try:
				offset = int(info["offset"])
			except KeyError:
				offset = 0
				pass
			net = DifferenceNetwork(self.top, self.trajs)
			c = net.deltaP(community_file, self.network, offset=offset)
			net.write_deltaP(c)		

		else:
			raise KeyError("Analysis type not found - Check .cfg file")

	def config_and_run_visual(self, _type):
		info = self.OPTIONS[_type]
		if _type == 'NETWORK':
			self.draw.draw_network()

		elif _type == 'COMMUNITIES':
			ncomms = int(info["number_of_communities"])
			self.draw.write_chimera_session(ncomms=ncomms)

		elif _type == 'SUBOPTIMALPATHS':
			try:
				minprob = float(info["minimum_cutoff"])
			except KeyError:
				minprob = 0.25
				pass

			self.draw.draw_paths(minprob=minprob)

		else:
			raise KeyError("Visual type not found - Check .cfg file")

	def execute(self):

		if self.visualize:
			from visualization import DrawObject
			self.draw = DrawObject(self.top, self.reference, network=self.network)
			for q in self.queue:
				self.config_and_run_visual(q)

		else:

			for q in self.queue:
				if q == 'NETWORK':
					self.config_and_run_network()
				elif q == 'COMMUNITIES' or q == 'SUBOPTIMALPATHS':
					self.config_and_run_analysis(q)
			
				elif q == 'DELTAP':
					assert self.network is not None, "No difference network found " + \
									"- Check .cfg file! You need to specify " + \
									" load_network = True and network_name = " + \
									" 'path/to/difference/network/file'"
					self.config_and_run_analysis(q)
				else:
					raise ValueError("Type not found - Check .cfg file")
		
