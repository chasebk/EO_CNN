#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:19, 25/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from models.root.hybrid.root_hybrid_deep_nets import RootHybridCnn
from mealpy.evolutionary_based import GA
from mealpy.swarm_based import WOA
from mealpy.physics_based import MVO, EO
from mealpy.bio_based import SBO
from mealpy.human_based import SSDO, LCBO
from mealpy.system_based import AEO


class GaCnn(RootHybridCnn):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, cnn_paras=None, ga_paras=None):
		RootHybridCnn.__init__(self, root_base_paras, root_hybrid_paras, cnn_paras)
		self.epoch = ga_paras["epoch"]
		self.pop_size = ga_paras["pop_size"]
		self.pc = ga_paras["pc"]
		self.pm = ga_paras["pm"]
		self.filename = "GA_CNN-" + root_hybrid_paras["paras_name"]

	def _training__(self):
		ga = GA.BaseGA(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size, self.pc, self.pm)
		self.solution, self.best_fit, self.loss_train = ga._train__()


class WoaCnn(RootHybridCnn):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, cnn_paras=None, woa_paras=None):
		RootHybridCnn.__init__(self, root_base_paras, root_hybrid_paras, cnn_paras)
		self.epoch = woa_paras["epoch"]
		self.pop_size = woa_paras["pop_size"]
		self.filename = "WOA_CNN-" + root_hybrid_paras["paras_name"]

	def _training__(self):
		md = WOA.BaseWOA(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size)
		self.solution, self.best_fit, self.loss_train = md._train__()


class MvoCnn(RootHybridCnn):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, cnn_paras=None, mvo_paras=None):
		RootHybridCnn.__init__(self, root_base_paras, root_hybrid_paras, cnn_paras)
		self.epoch = mvo_paras["epoch"]
		self.pop_size = mvo_paras["pop_size"]
		self.wep_minmax = mvo_paras["wep_minmax"]
		self.filename = "MVO_CNN-" + root_hybrid_paras["paras_name"]

	def _training__(self):
		md = MVO.BaseMVO(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size, self.wep_minmax)
		self.solution, self.best_fit, self.loss_train = md._train__()


class SboCnn(RootHybridCnn):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, cnn_paras=None, sbo_paras=None):
		RootHybridCnn.__init__(self, root_base_paras, root_hybrid_paras, cnn_paras)
		self.epoch = sbo_paras["epoch"]
		self.pop_size = sbo_paras["pop_size"]
		self.alpha = sbo_paras["alpha"]
		self.pm = sbo_paras["pm"]
		self.z = sbo_paras["z"]
		self.filename = "SBO_CNN-" + root_hybrid_paras["paras_name"]

	def _training__(self):
		md = SBO.BaseSBO(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size, self.alpha, self.pm, self.z)
		self.solution, self.best_fit, self.loss_train = md._train__()


class SsdoCnn(RootHybridCnn):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, cnn_paras=None, ssdo_paras=None):
		RootHybridCnn.__init__(self, root_base_paras, root_hybrid_paras, cnn_paras)
		self.epoch = ssdo_paras["epoch"]
		self.pop_size = ssdo_paras["pop_size"]
		self.filename = "SSDO_CNN-" + root_hybrid_paras["paras_name"]

	def _training__(self):
		md = SSDO.LevySSDO(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size)
		self.solution, self.best_fit, self.loss_train = md._train__()


class LcboCnn(RootHybridCnn):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, cnn_paras=None, lcbo_paras=None):
		RootHybridCnn.__init__(self, root_base_paras, root_hybrid_paras, cnn_paras)
		self.epoch = lcbo_paras["epoch"]
		self.pop_size = lcbo_paras["pop_size"]
		self.filename = "LCBO_CNN-" + root_hybrid_paras["paras_name"]

	def _training__(self):
		md = LCBO.LevyLCBO(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size)
		self.solution, self.best_fit, self.loss_train = md._train__()


class EoCnn(RootHybridCnn):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, cnn_paras=None, eo_paras=None):
		RootHybridCnn.__init__(self, root_base_paras, root_hybrid_paras, cnn_paras)
		self.epoch = eo_paras["epoch"]
		self.pop_size = eo_paras["pop_size"]
		self.filename = "EO_CNN-" + root_hybrid_paras["paras_name"]

	def _training__(self):
		md = EO.LevyEO(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size)
		self.solution, self.best_fit, self.loss_train = md._train__()


class AeoCnn(RootHybridCnn):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, cnn_paras=None, aeo_paras=None):
		RootHybridCnn.__init__(self, root_base_paras, root_hybrid_paras, cnn_paras)
		self.epoch = aeo_paras["epoch"]
		self.pop_size = aeo_paras["pop_size"]
		self.filename = "AEO_CNN-" + root_hybrid_paras["paras_name"]

	def _training__(self):
		md = AEO.LevyAEO(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size)
		self.solution, self.best_fit, self.loss_train = md._train__()
