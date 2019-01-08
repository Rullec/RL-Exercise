import numpy as np
from math import e, factorial

MAX_CAR_NUM = 20                # max capacity for rental point A and B
EARN_MONEY_RENT_PER_CAR = 10    # if you rent a car successfully, you can earn money for 10 euros
COST_MONEY_MOVE_PER_CAR = 2     # if you move a car, it costs 2 euros
MOVE_LIMIT = 5                  # the max num of moving cars in one night
DISCOUNT = 0.9

OPTIMAL = True
UNOPTIMAL = False

probability_lambda_equal_2 = [ 2**i / factorial(i) * e**(-2) for i in range(21)]
probability_lambda_equal_3 = [ 3**i / factorial(i) * e**(-3) for i in range(21)]
probability_lambda_equal_4 = [ 4**i / factorial(i) * e**(-4) for i in range(21)]

# initialization
# state-Value function for each state (a,b) and policy(the number of moving cars for particular state(a,b) overnight)
Value = np.zeros((MAX_CAR_NUM+1, MAX_CAR_NUM+1))
Policy = np.zeros((MAX_CAR_NUM+1, MAX_CAR_NUM+1))


def regular_state_num( state):
    """
    :param state: a number or a pair of number
    :return: an regulared number or a pair of number
    """
    if int == type(state):
        state = max(min(state, MAX_CAR_NUM), 0)
    elif list == type(state):
        assert 2==len(state)
        for i in range(0, 2):
            state[i] = max(min(state[i], MAX_CAR_NUM), 0)
    return state

def get_next_state_and_reward_pairs(now_state, action):
    '''
    :param now_state: a pair of numbers which range in [0, 20], represent the cars' number after that day
    :param action: a number from -5 to 5, the number of cars moved from B to A overnight
    :return: next_state, reward, probability
    '''
    print('get_next_state_and_reward for state: ' + str(now_state) + ', action: ' + str(action))
    reward = -2 * abs(action)
    now_state[0] = now_state[0] + action
    now_state[1] = now_state[1] - action
    now_state = regular_state_num(now_state)

    # decide point A's reward and corresponding pro
    A_state_reward_probability = np.zeros((MAX_CAR_NUM + 1, 2))

    # rent cars in A point
    A_rent_probability_reward = np.zeros((now_state[0]+1, 2))
    for i in range(now_state[0] + 1):
        if i != now_state[0]:
            A_rent_probability_reward[i, 0] = probability_lambda_equal_3[i]
        else:
            A_rent_probability_reward[i, 0] = 1 - sum(A_rent_probability_reward[:, 0])

        A_rent_probability_reward[i, 1] = i * EARN_MONEY_RENT_PER_CAR
    print('for point A, pro and reward matrix is : \n' + str(A_rent_probability_reward))

    MAX_BACK_NUM = MAX_CAR_NUM - now_state[0]
    A_back_probability = np.array(probability_lambda_equal_4)
    A_back_probability[-1] = 1 - np.sum(A_back_probability[0:MAX_CAR_NUM])
    print(A_back_probability)

    # 接下来要进行组合，组合成next_state - reward - probability 的列表

def policy_evaluation():
    print('policy evaluation')

def policy_improvement():
    print('policy improvement')
    return OPTIMAL
    return UNOPTIMAL

def jack_car_rental():
    iter = 0
    while True:

        # policy evaluation
        policy_evaluation()

        # policy improvement
        if(OPTIMAL == policy_improvement()):
            print('get optimal policy, state value func is \n' + str(Value))
            break
        iter = iter + 1
        print('iter ' + str(iter) + ':')

#jack_car_rental()

get_next_state_and_reward_pairs([1,2], 2)