import numpy as np
import matplotlib.pyplot as plt

ACTION_LEFT = 0
ACTION_RIGHT = 0
LEFT_LIMIT = -3
RIGHT_LIMIT = 3
CONST_BIAS_STATUS_2_INDEX = 2
VALUE_FUNC = []
VISIT_TIMES = []    # wait to init
ALPHA_MC= 0.001
ALPHA_TD = 0.001

def init_global():
    '''
    initial all of the global vars
    :return:
    '''
    global RIGHT_LIMIT, LEFT_LIMIT, VALUE_FUNC, VISIT_TIMES
    STATUS_NUM = RIGHT_LIMIT - LEFT_LIMIT - 1
    VALUE_FUNC = np.random.random([STATUS_NUM])
    VISIT_TIMES = np.ones([STATUS_NUM])

def generate_simulation_result_step():
    global  LEFT_LIMIT, RIGHT_LIMIT, CONST_BIAS_STATUS_2_INDEX
    status = 0
    while True:
        action = np.random.randint(0, 2)
        new_status = status
        if ACTION_LEFT == action:
            new_status = status - 1
        else:
            new_status = status + 1
        if LEFT_LIMIT == new_status:
            yield [status, 0]
            break
        if RIGHT_LIMIT == new_status:
            yield [status, 1]
            break
        yield [status, action, new_status]
        status = new_status

def generate_simulation_result_continuos():
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
        action = np.random.randint(0, 2)
        result.append([status, action])
        if ACTION_LEFT == action:
            status -= 1
        else:
            status += 1
    return result

def monte_carlo_method(episode):
    '''
    use monte carlo method to learn the optimal strategy of the random walk
    1. generate simulation result, [S A R] sequence as a list
    :param episode:
    :return:
    '''
    global ALPHA_MC, VALUE_FUNC, CONST_BIAS_STATUS_2_INDEX
    print('MC method for ' + str(episode) + ' episodes begin!')
    init_global()
    timer_1 = 0.0
    timer_0 = 0.0
    for i in range(episode):
        # policy evaluation, get value function corespondly
        result = generate_simulation_result_continuos()
        reward = result[-1][1]
        result = result[:-1]
        if 1 == reward:
            timer_1 = timer_1 + 1
        elif 0 == reward:
            timer_0 = timer_0 + 1
        bool_visited = np.zeros(RIGHT_LIMIT - LEFT_LIMIT - 1, dtype=bool)
        for ele in result:
            status = ele[0]
            # first-visit
            index = status + CONST_BIAS_STATUS_2_INDEX
            VALUE_FUNC[index] = VALUE_FUNC[index] + ALPHA_MC * (reward - VALUE_FUNC[index])
    print('VALUE func:' + str(VALUE_FUNC))
    # print(timer_0, timer_1)

def td_method(episode):
    '''
    use td method to learn the optimal strategy of the random walk
    :param episode:
    :return:
    '''
    global ALPHA_TD, VALUE_FUNC, CONST_BIAS_STATUS_2_INDEX
    print('td method for ' + str(episode) +' episodes begin!')
    init_global()
    for i in range(episode):
        # print('episode ' + str(i) + ' begin!')
        for ele in generate_simulation_result_step():
            if 2 == len(ele):
                now_status = ele[0] + CONST_BIAS_STATUS_2_INDEX
                reward = ele[1]
                VALUE_FUNC[now_status] = VALUE_FUNC[now_status] + ALPHA_TD * (
                            reward - VALUE_FUNC[now_status])
            elif 3 == len(ele):
                now_status = ele[0] + CONST_BIAS_STATUS_2_INDEX
                next_status = ele[2] + CONST_BIAS_STATUS_2_INDEX
                # print(now_status, next_status)
                VALUE_FUNC[now_status] = VALUE_FUNC[now_status] + ALPHA_TD * (
                                VALUE_FUNC[next_status] - VALUE_FUNC[now_status])

    print('VALUE func:' + str(VALUE_FUNC))

def main():
    episode = 10000
    monte_carlo_method(episode)
    td_method(episode)

init_global()
main()
