#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:11, 25/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%
"""
Link :  http://sci-hub.tw/10.1109/iccat.2013.6521977
        https://en.wikipedia.org/wiki/Laguerre_polynomials
"""

from numpy import where, maximum, exp
from numpy import tanh as nptanh


def elu(x, alpha=1):
    return where(x < 0, alpha * (exp(x) - 1), x)


def relu(x):
    return maximum(0, x)


def tanh(x):
    return nptanh(x)


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))
