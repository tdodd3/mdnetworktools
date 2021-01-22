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

import logging
import os
from datetime import datetime

class LOG(object):

	def __init__(self, logName):
		
		try:
			os.remove(logName)
		except OSError:
			pass
		
		logging.basicConfig(filename=logName, filemode='a',
					format='%(message)s', level=logging.INFO)
							
		self.inputheader = "#################### Input Parameters ####################"
		self.footer = "##########################################################"
		self.attrib = "################## Attributes ####################"
		self.objs = {0: "Dynamic Network...", 1: "Difference Network...",
				2: "Girvan-Newman...", 3: "SOAN...", 4: "Average Distances...",
				5: "Cross-Correlations...", 6: "Contacts...", 7: "Network...",
				8: "Consensus matrix...", 9: "Difference matrix..."}
		self.prefx = {0: "Initiating ", 1: "Building ", 2: "Computing "}
		self.NAME= "#                    MDNetworkTools                      #"
		
	def _startup(self):
		current_time = str(datetime.now())
		logging.info(current_time)
		logging.info(self.footer)
		logging.info(self.NAME)
		logging.info(self.footer)
		
	def _logit(self, m_type, params=None):
		message = self.prefx[m_type[0]] + self.objs[m_type[1]]
		logging.info(message)
		if params is not None:
			logging.info(self.inputheader)
			for p in params:
				key = p
				value = params[p]
				m = "{}: {}".format(key, value)
				logging.info(m)
			logging.info(self.footer)
		
	def _timing(self, m_type, duration):
		message = self.objs[m_type].split("...")[0] + " took {} seconds".format(duration)
		logging.info(message)
		
	def _generic(self, message):
		logging.info(message)
