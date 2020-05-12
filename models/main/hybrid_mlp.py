#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 17:28, 27/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from models.root.hybrid.root_hybrid_mlp import RootHybridMlp
from mealpy.evolutionary_based import GA, DE, FPA
from mealpy.swarm_based import PSO, WOA
from mealpy.physics_based import WDO, MVO, EO, NRO, HGSO
from time import time


class FpaMlp(RootHybridMlp):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, fpa_paras=None):
        RootHybridMlp.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = fpa_paras["epoch"]
        self.pop_size = fpa_paras["pop_size"]
        self.p = fpa_paras["p"]
        self.filename = root_hybrid_paras["paras_name"]

    def _training__(self):
        md_temp = FPA.BaseFPA(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size, self.p)
        self.solution, self.best_fit, self.loss_train = md_temp._train__()


class GaMlp(RootHybridMlp):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, ga_paras=None):
        RootHybridMlp.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = ga_paras["epoch"]
        self.pop_size = ga_paras["pop_size"]
        self.pc = ga_paras["pc"]
        self.pm = ga_paras["pm"]
        self.filename = root_hybrid_paras["paras_name"]

    def _training__(self):
        ga = GA.BaseGA(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size, self.pc, self.pm)
        self.solution, self.best_fit, self.loss_train = ga._train__()


class DeMlp(RootHybridMlp):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, de_paras=None):
        RootHybridMlp.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = de_paras["epoch"]
        self.pop_size = de_paras["pop_size"]
        self.wf = de_paras["wf"]
        self.cr = de_paras["cr"]
        self.filename = root_hybrid_paras["paras_name"]

    def _training__(self):
        de = DE.BaseDE(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size, self.wf, self.cr)
        self.solution, self.best_fit, self.loss_train = de._train__()


class PsoMlp(RootHybridMlp):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, pso_paras=None):
        RootHybridMlp.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = pso_paras["epoch"]
        self.pop_size = pso_paras["pop_size"]
        self.c1 = pso_paras["c1"]
        self.c2 = pso_paras["c2"]
        self.w_min = pso_paras["w_min"]
        self.w_max = pso_paras["w_max"]
        self.filename = root_hybrid_paras["paras_name"]

    def _training__(self):
        md = PSO.BasePSO(self._objective_function__, self.problem_size, self.domain_range, self.log,
                         self.epoch, self.pop_size, self.c1, self.c2, self.w_min, self.w_max)
        self.solution, self.best_fit, self.loss_train = md._train__()


class WoaMlp(RootHybridMlp):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, woa_paras=None):
        RootHybridMlp.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = woa_paras["epoch"]
        self.pop_size = woa_paras["pop_size"]
        self.filename = root_hybrid_paras["paras_name"]

    def _training__(self):
        md = WOA.BaseWOA(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size)
        self.solution, self.best_fit, self.loss_train = md._train__()


class WdoMlp(RootHybridMlp):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, wdo_paras=None):
        RootHybridMlp.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = wdo_paras["epoch"]
        self.pop_size = wdo_paras["pop_size"]
        self.RT = wdo_paras["RT"]
        self.g = wdo_paras["g"]
        self.alp = wdo_paras["alp"]
        self.c = wdo_paras["c"]
        self.max_v = wdo_paras["max_v"]
        self.filename = root_hybrid_paras["paras_name"]

    def _training__(self):
        md = WDO.BaseWDO(self._objective_function__, self.problem_size, self.domain_range, self.log,
                         self.epoch, self.pop_size, self.RT, self.g, self.alp, self.c, self.max_v)
        self.solution, self.best_fit, self.loss_train = md._train__()


class MvoMlp(RootHybridMlp):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, mvo_paras=None):
        RootHybridMlp.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = mvo_paras["epoch"]
        self.pop_size = mvo_paras["pop_size"]
        self.wep_minmax = mvo_paras["wep_minmax"]
        self.filename = root_hybrid_paras["paras_name"]

    def _training__(self):
        md = MVO.BaseMVO(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size, self.wep_minmax)
        self.solution, self.best_fit, self.loss_train = md._train__()


class EoMlp(RootHybridMlp):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, eo_paras=None):
        RootHybridMlp.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = eo_paras["epoch"]
        self.pop_size = eo_paras["pop_size"]
        self.filename = root_hybrid_paras["paras_name"]

    def _training__(self):
        md = EO.LevyEO(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size)
        self.solution, self.best_fit, self.loss_train = md._train__()


class NroMlp(RootHybridMlp):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, nro_paras=None):
        RootHybridMlp.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = nro_paras["epoch"]
        self.pop_size = nro_paras["pop_size"]
        self.filename = root_hybrid_paras["paras_name"]

    def _training__(self):
        md = NRO.BaseNRO(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size)
        self.solution, self.best_fit, self.loss_train = md._train__()


class HgsoMlp(RootHybridMlp):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, hgso_paras=None):
        RootHybridMlp.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = hgso_paras["epoch"]
        self.pop_size = hgso_paras["pop_size"]
        self.n_clusters = hgso_paras["n_clusters"]
        self.filename = root_hybrid_paras["paras_name"]

    def _training__(self):
        md = HGSO.LevyHGSO(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size, self.n_clusters)
        self.solution, self.best_fit, self.loss_train = md._train__()


class PhysicsMlp(RootHybridMlp):
    def __init__(self, root_base_paras=None, root_hybrid_paras=None, algorithm_paras=None):
        RootHybridMlp.__init__(self, root_base_paras, root_hybrid_paras)
        self.epoch = algorithm_paras["epoch"]
        self.pop_size = algorithm_paras["pop_size"]
        self.RT = algorithm_paras["RT"]
        self.g = algorithm_paras["g"]
        self.alp = algorithm_paras["alp"]
        self.c = algorithm_paras["c"]
        self.max_v = algorithm_paras["max_v"]

        self.wep_minmax = algorithm_paras["wep_minmax"]

        self.n_clusters = algorithm_paras["n_clusters"]

        self.filename = root_hybrid_paras["paras_name"]

    def _training__(self):
        md1 = WDO.BaseWDO(self._objective_function__, self.problem_size, self.domain_range, self.log,
                         self.epoch, self.pop_size, self.RT, self.g, self.alp, self.c, self.max_v)
        self.solution1, self.best_fit1, self.loss_train1 = md1._train__()

        md2 = MVO.BaseMVO(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size, self.wep_minmax)
        self.solution2, self.best_fit2, self.loss_train = md2._train__()

        md3 = EO.LevyEO(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size)
        self.solution3, self.best_fit3, self.loss_train3 = md3._train__()

        md4 = NRO.BaseNRO(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size)
        self.solution4, self.best_fit4, self.loss_train4 = md4._train__()

        md5 = HGSO.LevyHGSO(self._objective_function__, self.problem_size, self.domain_range, self.log, self.epoch, self.pop_size, self.n_clusters)
        self.solution5, self.best_fit5, self.loss_train5 = md5._train__()


    def _running__(self):
        self.time_system = time()
        self._processing__()
        self._setting__()
        self.time_total_train = time()
        self._training__()

        self.model1 = self._get_model__(self.solution1)
        self.model2 = self._get_model__(self.solution2)
        self.model3 = self._get_model__(self.solution3)
        self.model4 = self._get_model__(self.solution4)
        self.model5 = self._get_model__(self.solution5)

        self.time_total_train = round(time() - self.time_total_train, 4)
        self.time_epoch = round(self.time_total_train / self.epoch, 4)
        self.time_predict = time()

        self.model = self.model1
        y_true_unscaled1, y_pred_unscaled1, y_true_scaled1, y_pred_scaled1 = self._forecasting__()
        self.model = self.model2
        y_true_unscaled2, y_pred_unscaled2, y_true_scaled2, y_pred_scaled2 = self._forecasting__()
        self.model = self.model3
        y_true_unscaled3, y_pred_unscaled3, y_true_scaled3, y_pred_scaled3 = self._forecasting__()
        self.model = self.model4
        y_true_unscaled4, y_pred_unscaled4, y_true_scaled4, y_pred_scaled4 = self._forecasting__()
        self.model = self.model5
        y_true_unscaled5, y_pred_unscaled5, y_true_scaled5, y_pred_scaled5 = self._forecasting__()

        y_pred = (y_pred_scaled1 + y_pred_scaled2 + y_pred_scaled3 + y_pred_scaled4 + y_pred_scaled5) / 5
        y_pred_unscaled = self.time_series._inverse_scaling__(y_pred, scale_type=self.scaling)

        self.time_predict = round(time() - self.time_predict, 8)
        self.time_system = round(time() - self.time_system, 4)
        self._save_results__(y_true_unscaled1, y_pred_unscaled, y_true_scaled1, y_pred, self.loss_train)
