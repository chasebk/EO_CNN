#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:11, 25/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%


"""
Link :  http://sci-hub.tw/10.1109/iccat.2013.6521977
        https://en.wikipedia.org/wiki/Laguerre_polynomials
"""

from numpy import where, maximum, power, multiply, concatenate, pi, sin, exp, cos
from numpy import tanh as nptanh


def itself(x):
	return x

def elu(x, alpha=1):
	return where(x < 0, alpha * (exp(x) - 1), x)

def relu(x):
	return maximum(0, x)

def tanh(x):
	return nptanh(x)

def sigmoid(x):
	return 1.0 / (1.0 + exp(-x))

def derivative_self(x):
	return 1

def derivative_elu(x, alpha=1):
	return where(x < 0, x + alpha, 1)


def derivative_relu(x):
	return where(x < 0, 0, 1)

def derivative_tanh(x):
	return 1 - power(x, 2)

def derivative_sigmoid(x):
	return multiply(x, 1 - x)


def expand_chebyshev(x):
	x1 = x
	x2 = 2 * power(x, 2) - 1
	x3 = 4 * power(x, 3) - 3 * x
	x4 = 8 * power(x, 4) - 8 * power(x, 2) + 1
	x5 = 16 * power(x, 5) - 20 * power(x, 3) + 5 * x
	return concatenate((x1, x2, x3, x4, x5), axis=1)

def expand_legendre(x):
	x1 = x
	x2 = 1 / 2 * (3 * power(x, 2) - 1)
	x3 = 1 / 2 * (5 * power(x, 3) - 3 * x)
	x4 = 1 / 8 * (35 * power(x, 4) - 30 * power(x, 2) + 3)
	x5 = 1 / 40 * (9 * power(x, 5) - 350 * power(x, 3) + 75 * x)
	return concatenate((x1, x2, x3, x4, x5), axis=1)

def expand_laguerre(x):
	x1 = -x + 1
	x2 = 1 / 2 * (power(x, 2) - 4 * x + 2)
	x3 = 1 / 6 * (-power(x, 3) + 9 * power(x, 2) - 18 * x + 6)
	x4 = 1 / 24 * (power(x, 4) - 16 * power(x, 3) + 72 * power(x, 2) - 96 * x + 24)
	x5 = 1 / 120 * (-power(x, 5) + 25 * power(x, 4) - 200 * power(x, 3) + 600 * power(x, 2) - 600 * x + 120)
	return concatenate((x1, x2, x3, x4, x5), axis=1)

def expand_power(x):
	x1 = x
	x2 = x1 + power(x, 2)
	x3 = x2 + power(x, 3)
	x4 = x3 + power(x, 4)
	x5 = x4 + power(x, 5)
	return concatenate((x1, x2, x3, x4, x5), axis=1)

def expand_trigonometric(x):
	x1 = x
	x2 = sin(pi * x) + cos(pi * x)
	x3 = sin(2 * pi * x) + cos(2 * pi * x)
	x4 = sin(3 * pi * x) + cos(3 * pi * x)
	x5 = sin(4 * pi * x) + cos(4 * pi * x)
	return concatenate((x1, x2, x3, x4, x5), axis=1)