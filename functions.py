"""
Author: Wiktor Kulesza
Date: 10.03.2021r.
"""
import numpy as np
import math


# Problem 1
# f(x)
def function1(x):
    return pow(x, 2)


# gradient
def gradient1(x):
    return 2 * x


# Problem 2
# f(x)
def function2(x, y):
    return np.subtract(np.add(np.square(x), np.square(y)), 5 * np.cos(10 * np.sqrt(np.add(np.square(x), np.square(y)))))


# gradient
def gradient2(x):
    gradient = np.array([x[0] * (2 + function_g(x)), x[1] * (2 + function_g(x))])
    return gradient


def function_g(x):
    return 50 * math.sin(10 * math.sqrt(x[0] ** 2 + x[1] ** 2)) / math.sqrt(x[0] ** 2 + x[1] ** 2)