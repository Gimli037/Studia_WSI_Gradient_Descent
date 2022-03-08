"""
Author: Wiktor Kulesza
Date: 10.03.2021r.
"""
import numpy as np
import sys


def gradient_descent(gradient, start_point, learning_rate, num_of_iter):
    trace = np.array([start_point])
    curr_point = start_point
    for _ in range(num_of_iter):
        step = -1 * learning_rate * gradient(curr_point)
        if np.all((abs(step) < sys.float_info.epsilon)):
            break
        curr_point += step
        trace = np.append(trace, [curr_point], axis=0)
    return trace



