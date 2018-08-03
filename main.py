from numpy import *

def compute_error_for_line_given_points(c, m, points):
    # initialize it at 0
    total_error = 0.0
    #for every point
    for i  in range(0, len(points)):#
        # get the x value
        x = points[i, 0]
        # get the y value
        y = points[i, 1]
        # get the difference, square it and add it to the total
        total_error += (y - (m * x + c)) ** 2

    # get the average and return
    return total_error / float(len(points))

def gradient_descent_runner(points, starting_c, starting_m, learning_rate, num_iteration):
    # starting c and m
    c = starting_c
    m = starting_m

    # gradient descent
    for i in range(num_iteration):
        # update c and m with more accurate c and m by performing this gradient step
        c, m = step_gradient(c, m, array(points), learning_rate)

    return [c, m]

def step_gradient(c_current, m_current, points, learning_rate):

    c_gradient = 0
    m_gradient = 0

    N = float(len(points))

    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # direction wrt c and m computing partial derivatives of our error function
        c_gradient += -(2/N) * (y - ((m_current * x) + c_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + c_current))

    # update our c and m values using our partial derivatives
    new_c = c_current - (learning_rate * c_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return [new_c, new_m]

def run():
    #Step 1 - collect data
    points = genfromtxt('data.csv', delimiter=',')

    #Step 2 - define our hyperparameters
    learning_rate = 0.0001
    # y = mx + c (slope formula)
    initial_c = 0
    initial_m = 0
    num_iteration = 1000

    #Step 3 - train our model
    print('starting gradient descent at c = {0}, m = {1}, error = {2}'.format(initial_c, initial_m, compute_error_for_line_given_points(initial_c, initial_m, points)))
    [c, m] = gradient_descent_runner(points, initial_c, initial_m, learning_rate, num_iteration)
    print('After {0} iterations, ending gradient descent at c = {1}, m = {2}, error = {3}'.format(num_iteration, c, m, compute_error_for_line_given_points(c, m, points)))
    


if __name__ == '__main__':
    run()