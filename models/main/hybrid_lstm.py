#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 19:00, 25/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from models.root.hybrid.root_hybrid_deep_nets import RootHybridLstm
from mealpy.evolutionary_based import GA, DE, FPA
from mealpy.swarm_based import PSO, WOA
from mealpy.physics_based import WDO, MVO, EO, NRO, HGSO, ASO


class FpaLstm(RootHybridLstm):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, algo_paras=None):
		RootHybridLstm.__init__(self, root_base_paras, root_hybrid_paras)
		self.epoch = algo_paras["epoch"]
		self.pop_size = algo_paras["pop_size"]
		self.p = algo_paras["p"]
		self.filename = root_hybrid_paras["paras_name"]

	def _training__(self):
		md_temp = FPA.BaseFPA(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size, self.p)
		self.solution, self.best_fit, self.loss_train = md_temp._train__()


class GaLstm(RootHybridLstm):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, algo_paras=None):
		RootHybridLstm.__init__(self, root_base_paras, root_hybrid_paras)
		self.epoch = algo_paras["epoch"]
		self.pop_size = algo_paras["pop_size"]
		self.pc = algo_paras["pc"]
		self.pm = algo_paras["pm"]
		self.filename = root_hybrid_paras["paras_name"]

	def _training__(self):
		ga = GA.BaseGA(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size, self.pc, self.pm)
		self.solution, self.best_fit, self.loss_train = ga._train__()


class DeLstm(RootHybridLstm):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, algo_paras=None):
		RootHybridLstm.__init__(self, root_base_paras, root_hybrid_paras)
		self.epoch = algo_paras["epoch"]
		self.pop_size = algo_paras["pop_size"]
		self.wf = algo_paras["wf"]
		self.cr = algo_paras["cr"]
		self.filename = root_hybrid_paras["paras_name"]

	def _training__(self):
		de = DE.BaseDE(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size, self.wf, self.cr)
		self.solution, self.best_fit, self.loss_train = de._train__()


class PsoLstm(RootHybridLstm):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, algo_paras=None):
		RootHybridLstm.__init__(self, root_base_paras, root_hybrid_paras)
		self.epoch = algo_paras["epoch"]
		self.pop_size = algo_paras["pop_size"]
		self.c1 = algo_paras["c1"]
		self.c2 = algo_paras["c2"]
		self.w_min = algo_paras["w_min"]
		self.w_max = algo_paras["w_max"]
		self.filename = root_hybrid_paras["paras_name"]

	def _training__(self):
		md = PSO.BasePSO(self._objective_function__, self.problem_size, self.domain_range, self.log,
		                 self.epoch, self.pop_size, self.c1, self.c2, self.w_min, self.w_max)
		self.solution, self.best_fit, self.loss_train = md._train__()


class WoaLstm(RootHybridLstm):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, algo_paras=None):
		RootHybridLstm.__init__(self, root_base_paras, root_hybrid_paras)
		self.epoch = algo_paras["epoch"]
		self.pop_size = algo_paras["pop_size"]
		self.filename = root_hybrid_paras["paras_name"]

	def _training__(self):
		md = WOA.BaseWOA(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size)
		self.solution, self.best_fit, self.loss_train = md._train__()


class WdoLstm(RootHybridLstm):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, algo_paras=None):
		RootHybridLstm.__init__(self, root_base_paras, root_hybrid_paras)
		self.epoch = algo_paras["epoch"]
		self.pop_size = algo_paras["pop_size"]
		self.RT = algo_paras["RT"]
		self.g = algo_paras["g"]
		self.alp = algo_paras["alp"]
		self.c = algo_paras["c"]
		self.max_v = algo_paras["max_v"]
		self.filename = root_hybrid_paras["paras_name"]

	def _training__(self):
		md = WDO.BaseWDO(self._objective_function__, self.problem_size, self.domain_range, self.log,
		                 self.epoch, self.pop_size, self.RT, self.g, self.alp, self.c, self.max_v)
		self.solution, self.best_fit, self.loss_train = md._train__()


class MvoLstm(RootHybridLstm):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, algo_paras=None):
		RootHybridLstm.__init__(self, root_base_paras, root_hybrid_paras)
		self.epoch = algo_paras["epoch"]
		self.pop_size = algo_paras["pop_size"]
		self.wep_minmax = algo_paras["wep_minmax"]
		self.filename = root_hybrid_paras["paras_name"]

	def _training__(self):
		md = MVO.BaseMVO(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size, self.wep_minmax)
		self.solution, self.best_fit, self.loss_train = md._train__()


class EoLstm(RootHybridLstm):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, algo_paras=None):
		RootHybridLstm.__init__(self, root_base_paras, root_hybrid_paras)
		self.epoch = algo_paras["epoch"]
		self.pop_size = algo_paras["pop_size"]
		self.filename = root_hybrid_paras["paras_name"]

	def _training__(self):
		md = EO.LevyEO(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size)
		self.solution, self.best_fit, self.loss_train = md._train__()


class NroLstm(RootHybridLstm):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, algo_paras=None):
		RootHybridLstm.__init__(self, root_base_paras, root_hybrid_paras)
		self.epoch = algo_paras["epoch"]
		self.pop_size = algo_paras["pop_size"]
		self.filename = root_hybrid_paras["paras_name"]

	def _training__(self):
		md = NRO.BaseNRO(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size)
		self.solution, self.best_fit, self.loss_train = md._train__()


class HgsoLstm(RootHybridLstm):
	def __init__(self, root_base_paras=None, root_hybrid_paras=None, algo_paras=None):
		RootHybridLstm.__init__(self, root_base_paras, root_hybrid_paras)
		self.epoch = algo_paras["epoch"]
		self.pop_size = algo_paras["pop_size"]
		self.n_clusters = algo_paras["n_clusters"]
		self.filename = root_hybrid_paras["paras_name"]

	def _training__(self):
		md = HGSO.LevyHGSO(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size, self.n_clusters)
		self.solution, self.best_fit, self.loss_train = md._train__()
