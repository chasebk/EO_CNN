#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 19:00, 25/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from models.root.hybrid.root_hybrid_deep_nets import RootHybridLstm
from mealpy.evolutionary_based import GA, DE
from mealpy.swarm_based import PSO, WOA
from mealpy.physics_based import WDO, MVO, EO


class GaLstm(RootHybridLstm):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, ga_paras=None):
		RootHybridLstm.__init__(self, root_base_paras, root_hybrid_paras)
		self.epoch = ga_paras["epoch"]
		self.pop_size = ga_paras["pop_size"]
		self.pc = ga_paras["pc"]
		self.pm = ga_paras["pm"]
		self.filename = "GA_LSTM-" + root_hybrid_paras["paras_name"]

	def _training__(self):
		ga = GA.BaseGA(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size, self.pc, self.pm)
		self.solution, self.best_fit, self.loss_train = ga._train__()


class DeLstm(RootHybridLstm):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, de_paras=None):
		RootHybridLstm.__init__(self, root_base_paras, root_hybrid_paras)
		self.epoch = de_paras["epoch"]
		self.pop_size = de_paras["pop_size"]
		self.wf = de_paras["wf"]
		self.cr = de_paras["cr"]
		self.filename = "DE_LSTM-" + root_hybrid_paras["paras_name"]

	def _training__(self):
		de = DE.BaseDE(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size, self.wf, self.cr)
		self.solution, self.best_fit, self.loss_train = de._train__()


class PsoLstm(RootHybridLstm):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, pso_paras=None):
		RootHybridLstm.__init__(self, root_base_paras, root_hybrid_paras)
		self.epoch = pso_paras["epoch"]
		self.pop_size = pso_paras["pop_size"]
		self.c1 = pso_paras["c1"]
		self.c2 = pso_paras["c2"]
		self.w_min = pso_paras["w_min"]
		self.w_max = pso_paras["w_max"]
		self.filename = "PSO_LSTM-" + root_hybrid_paras["paras_name"]

	def _training__(self):
		md = PSO.BasePSO(self._objective_function__, self.problem_size, self.domain_range, self.log,
		                 self.epoch, self.pop_size, self.c1, self.c2, self.w_min, self.w_max)
		self.solution, self.best_fit, self.loss_train = md._train__()


class WoaLstm(RootHybridLstm):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, woa_paras=None):
		RootHybridLstm.__init__(self, root_base_paras, root_hybrid_paras)
		self.epoch = woa_paras["epoch"]
		self.pop_size = woa_paras["pop_size"]
		self.filename = "WOA_LSTM-" + root_hybrid_paras["paras_name"]

	def _training__(self):
		md = WOA.BaseWOA(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size)
		self.solution, self.best_fit, self.loss_train = md._train__()


class WdoLstm(RootHybridLstm):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, wdo_paras=None):
		RootHybridLstm.__init__(self, root_base_paras, root_hybrid_paras)
		self.epoch = wdo_paras["epoch"]
		self.pop_size = wdo_paras["pop_size"]
		self.RT = wdo_paras["RT"]
		self.g = wdo_paras["g"]
		self.alp = wdo_paras["alp"]
		self.c = wdo_paras["c"]
		self.max_v = wdo_paras["max_v"]
		self.filename = "WDO_LSTM-" + root_hybrid_paras["paras_name"]

	def _training__(self):
		md = WDO.BaseWDO(self._objective_function__, self.problem_size, self.domain_range, self.log,
		                 self.epoch, self.pop_size, self.RT, self.g, self.alp, self.c, self.max_v)
		self.solution, self.best_fit, self.loss_train = md._train__()


class MvoLstm(RootHybridLstm):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, mvo_paras=None):
		RootHybridLstm.__init__(self, root_base_paras, root_hybrid_paras)
		self.epoch = mvo_paras["epoch"]
		self.pop_size = mvo_paras["pop_size"]
		self.wep_minmax = mvo_paras["wep_minmax"]
		self.filename = "WVO_LSTM-" + root_hybrid_paras["paras_name"]

	def _training__(self):
		md = MVO.BaseMVO(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size, self.wep_minmax)
		self.solution, self.best_fit, self.loss_train = md._train__()


class EoLstm(RootHybridLstm):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, eo_paras=None):
		RootHybridLstm.__init__(self, root_base_paras, root_hybrid_paras)
		self.epoch = eo_paras["epoch"]
		self.pop_size = eo_paras["pop_size"]
		self.filename = "EO_LSTM-" + root_hybrid_paras["paras_name"]

	def _training__(self):
		md = EO.BaseEO(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size)
		self.solution, self.best_fit, self.loss_train = md._train__()