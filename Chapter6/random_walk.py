import numpy as np
import random

ACTION_LEFT = 0
ACTION_RIGHT = 0
LEFT_LIMIT = -3
RIGHT_LIMIT = 3
CONST_BIAS_STATUS_2_INDEX = 2
POLICY = []
VALUE_FUNC = []

def init_global():
    '''
    initial all of the global vars
    :return:
    '''
    global RIGHT_LIMIT, LEFT_LIMIT, POLICY, VALUE_FUNC
    STATUS_NUM = RIGHT_LIMIT - LEFT_LIMIT - 1
    POLICY = np.array([ ACTION_LEFT for i in range(STATUS_NUM)])
    VALUE_FUNC = np.random.rand(STATUS_NUM, 2)

def generate_simulation_result():
    '''
    :return: [[status0, action0], ... , [status0, action0]]
    '''
    global  LEFT_LIMIT, RIGHT_LIMIT, CONST_BIAS_STATUS_2_INDEX
    result = []
    status = 0
    while True:
        if LEFT_LIMIT == status:
            result.append([LEFT_LIMIT, 0])
            break
        if RIGHT_LIMIT == status:
            result.append([RIGHT_LIMIT, 1])
            break
        action = POLICY[status + CONST_BIAS_STATUS_2_INDEX]
        result.append([status, action])
        if ACTION_LEFT == action:
            status -= 1
        else:
            status += 1
    print(status)
    return result

def monte_carlo_method(episode):
    '''
    use monte carlo method to learn the optimal strategy of the random walk
    1. generate simulation result, [S A R] sequence as a list
    :param episode:
    :return:
    '''
    print('MC method for' + str(episode) + ' episodes begin!')
    init_global()
    for i in range(episode):
        # policy evaluation
        # policy improvement

def td_method(episode):
    '''
    use td method to learn the optimal strategy of the random walk
    :param episode:
    :return:
    '''
    print('td method for ' +str(episode) +' episodes begin!')
    init_global()
    for i in range(episode):
        pass

def main():

    monte_carlo_method()
    # td_method()

init_global()
a = generate_simulation_result()
print(a)