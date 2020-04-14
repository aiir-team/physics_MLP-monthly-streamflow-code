#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 23:50, 29/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from mealpy.evolutionary_based import GA, DE
from mealpy.swarm_based import PSO, WOA
from mealpy.physics_based import WDO, MVO, EO
from numpy.random import uniform

def _quartic__(solution=None):
    """
    Class: multimodal, non-convex, differentiable, separable, continuous, random
    Global: one global minimum fx = 0 + random, at (0, ...,0)
    Link: http://benchmarkfcns.xyz/benchmarkfcns/quarticfcn.html
    @param solution: A numpy array with x_i in [-1.28, 1.28]
    @return: fx
    """
    d = len(solution)
    result = 0
    for i in range(0, d):
        result += (i + 1) * solution[i] ** 4
    return result + uniform(0, 1)

problem_size = 300
domain_range = (-1, 1)
log = True
epoch = 1000
pop_size = 50

ga = GA.BaseGA(_quartic__, problem_size, domain_range, log, epoch, pop_size)
ga._train__()

eo = EO.BaseEO(_quartic__, problem_size, domain_range, log, epoch, pop_size)
eo._train__()