"""
Author: Wiktor Kulesza
Date: 10.03.2021r.
"""
import numpy as np
import functions as f
import gradient_descent as gd
import matplotlib.pyplot as plt

def main():
    # Checking different learning rates values and their impact on trace of the gradient (2D function)
    # Creating different learning rates
    learning_rates = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.9, 1.01])
    x = np.arange(-10.0, 0, 0.01) + np.arange(0.01, 10.01, 0.01)
    y = f.function1(x)
    for alpha in learning_rates:
        # Setting starting point
        starting_point = np.array([5.0])
        gradient_trace = gd.gradient_descent(f.gradient1, starting_point, alpha, num_of_iter=100)
        gradient_trace = np.concatenate((gradient_trace, f.function1(gradient_trace)), axis=1)
        fig, ax = plt.subplots()
        ax.set_title(f'Learning rate = {alpha}')
        plt.xlabel(f'Starting point = {gradient_trace[0][0]} Ending point = {gradient_trace[-1][0]}')
        trace_line = plt.plot(gradient_trace[:, 0], gradient_trace[:, 1], 'o--', label='Gradient trace',
                              linewidth=1, markersize=2)
        function_line = plt.plot(x, y, '-', label='Function = x^2', linewidth=1, markersize=2)
        plt.legend(loc="upper right")
        plt.show()

    # Checking different learning rates values and their impact on trace of the gradient (3D function)
    # Creating different learning rates
    learning_rates = np.array([0.001, 0.003, 0.005, 0.008, 0.01])
    x = np.arange(-2.0, 0, 0.01) + np.arange(0.01, 2.01, 0.01)
    y = np.arange(-2.0, 0, 0.01) + np.arange(0.01, 2.01, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = f.function2(X, Y)
    for alpha in learning_rates:
        # Setting starting point
        starting_point = np.array([1.0, -1.0])
        gradient_trace = gd.gradient_descent(f.gradient2, starting_point, alpha, num_of_iter=100)
        gradient_trace = np.column_stack((gradient_trace, f.function2(gradient_trace[:, 0], gradient_trace[:, 1])))
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        ax.set_title(f'Learning rate = {alpha}')
        plt.xlabel(f'Starting point = {gradient_trace[0][0:2]} Ending point = {gradient_trace[-1][0:2]}')
        cf = ax.contourf(X, Y, Z)
        fig.colorbar(cf, ax=ax)
        trace_line = plt.plot(gradient_trace[:, 0], gradient_trace[:, 1], 'o--', color='yellow',
                              label='Gradient trace', linewidth=1, markersize=2)
        plt.show()
        print(gradient_trace[-1][0:2])

    # Checking different starting point values and their impact on trace of the gradient (2D function)
    # Creating different starting points
    starting_points = np.array([-6.0, 0.0, 6.0])
    # Setting learning rate to 0.03 (best value based on previous contour graphs)
    learning_rate = 0.1
    x = np.arange(-8.0, 0, 0.01) + np.arange(0.01, 8.01, 0.01)
    y = f.function1(x)
    for starting_point in starting_points:
        fig, ax = plt.subplots()
        gradient_trace = gd.gradient_descent(f.gradient1, starting_point, learning_rate, num_of_iter=100)
        plt.plot(gradient_trace, f.function1(gradient_trace), 'o--', label='Gradient trace', linewidth=1, markersize=2)
        plt.plot(x, y, '-', label='Function = x^2', linewidth=1, markersize=2)
        ax.set_title(f'Starting point = {gradient_trace[0]}')
        plt.xlabel(f'learning rate = {learning_rate} Ending point = {gradient_trace[-1]}')
        plt.legend(loc="upper right")
        plt.show()

    # Checking different starting point values and their impact on trace of the gradient (3D function)
    # Creating different starting points
    starting_points = np.array([[-2.1, -2.1], [1.1, 1.1],
                                [-0.5, -0.8], [0.5, 0.5]])
    # Setting learning rate to 0.003 (best value based on previous contour graphs)
    learning_rate = 0.003
    x = np.arange(-2.5, 0, 0.01) + np.arange(0.01, 2.51, 0.01)
    y = np.arange(-2.5, 0, 0.01) + np.arange(0.01, 2.51, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = f.function2(X, Y)
    for starting_point in starting_points:
        gradient_trace = gd.gradient_descent(f.gradient2, starting_point, learning_rate, num_of_iter=50)
        gradient_trace = np.column_stack((gradient_trace, f.function2(gradient_trace[:, 0], gradient_trace[:, 1])))
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        plt.xlabel(f'Learning rate = {learning_rate} Ending point = {gradient_trace[-1][0:2]}')
        ax.set_title(f'Starting point = {gradient_trace[0][0:2]}')
        cf = ax.contourf(X, Y, Z)
        fig.colorbar(cf, ax=ax)
        trace_line = plt.plot(gradient_trace[:, 0], gradient_trace[:, 1], 'o--', color='yellow',
                              label='Gradient trace', linewidth=1, markersize=2)
        plt.show()

    # Checking objective function based on number of iterations of Gradient function (checking on problem 1 function)
    # Setting starting point to -6.0
    starting_point = -6.0
    # Setting learning rate to 0.03 (best value based on previous contour graphs)
    learning_rate = 0.1
    x = np.arange(-8.0, 8.01, 0.01)
    y = f.function1(x)
    fig, ax = plt.subplots()
    gradient_trace = gd.gradient_descent(f.gradient1, starting_point, learning_rate, num_of_iter=10000)
    num_of_iterations = gradient_trace.shape[0]
    plt.plot(np.arange(0, num_of_iterations, 1), f.function1(gradient_trace), 'o--',
             label='Gradient trace', linewidth=1, markersize=2)
    plt.xlabel(f'Num of iterations')
    plt.ylabel(f'Objective function value')
    ax.set_title(f'Starting point = {starting_point} Learning rate = {learning_rate}')
    plt.legend(loc="upper right")
    plt.show()


if __name__ == "__main__":
    main()
