#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 20:13, 25/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from models.root.hybrid.root_hybrid_deep_nets import RootHybridRnn
from mealpy.evolutionary_based import GA


class GaRnn(RootHybridRnn):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, ga_paras=None):
		RootHybridRnn.__init__(self, root_base_paras, root_hybrid_paras)
		self.epoch = ga_paras["epoch"]
		self.pop_size = ga_paras["pop_size"]
		self.pc = ga_paras["pc"]
		self.pm = ga_paras["pm"]
		self.filename = "GA_RNN-" + root_hybrid_paras["paras_name"]

	def _training__(self):
		ga = GA.BaseGA(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size, self.pc, self.pm)
		self.solution, self.best_fit, self.loss_train = ga._train__()