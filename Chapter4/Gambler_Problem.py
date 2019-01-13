# 22:57 2019/01/13 IST office
import numpy as np
import matplotlib.pyplot as plt

CONVERGENCE_LIMIT = 10e-15
MAX_MONEY = 100
MIN_FLOAT = -10e15

P_H = 0.25
value = np.zeros(MAX_MONEY + 1)
old_value = np.zeros(MAX_MONEY + 1)
policy = np.zeros(MAX_MONEY + 1)


def legal_assert(state, action):
    assert state>0 and state< MAX_MONEY, ("state illegal:" + str(state))
    assert action >=1 and action <= min(state, MAX_MONEY-state), ("action illegal, [state, action] : [%d, %d]" % (state, action) )

def get_next_state_probability_reward(cur_money, action):
    '''
    calculate a list contained  next state's transimission probability and reward
    :param cur_money:
    :param action:
    :return: [ nextstate, reward]
    '''
    legal_assert(cur_money, action)

    next_state_probability_reward = np.zeros((2, 3))
    # if the coin comes up heads
    next_state_probability_reward[0, 0] = cur_money + action
    next_state_probability_reward[0, 1] = P_H
    next_state_probability_reward[0, 2] = 0

    # if the coin is tails
    next_state_probability_reward[1, 0] = cur_money - action
    next_state_probability_reward[1, 1] = 1 - P_H
    next_state_probability_reward[1, 2] = 0

    return next_state_probability_reward

def value_iteration():
    '''

    :return:
    '''

    for cur_money in range(1, MAX_MONEY ):
        max_value = MIN_FLOAT
        for action in range(1, min(cur_money, MAX_MONEY - cur_money) + 1):
            legal_assert(cur_money, action)

            # calculate new value for state s: v(s) = argmax p(s',r|s,a)[r + v(s')]
            next_state_probability_reward = get_next_state_probability_reward(cur_money, action)
            temp_sum = 0
            for i in range(2):
                next_state = int(next_state_probability_reward[i, 0])
                pro = next_state_probability_reward[i, 1]
                reward = next_state_probability_reward[i, 2]
                temp_sum = temp_sum + pro * (reward + value[next_state])

            # value update
            max_value = max(temp_sum, max_value)
        value[cur_money] = max_value


def calculate_optimal_policy():
    '''

    :return:
    '''
    for state in range(1, MAX_MONEY):
        optimal_policy = 0
        value_array = []
        for action in range(1, min(state, MAX_MONEY - state) + 1):
            legal_assert(state, action)
            next_state_probability_reward = get_next_state_probability_reward(state, action)
            cur_action_value = 0
            for i in range(2):
                next_state = int(next_state_probability_reward[i, 0])
                pro = next_state_probability_reward[i, 1]
                reward = next_state_probability_reward[i, 2]
                cur_action_value = cur_action_value + pro * (reward + value[next_state])

            value_array.append(cur_action_value)

            # ATTENTION! BECAUSE OF THE NUMERICAL DIFFERENCE BETWEEN ITEMS IN THIS ARRAY, SO WE MUST CALL NP.ROUND
            policy[state] = np.argmax(np.round(value_array, 10)) + 1


    plt.plot(policy[1:MAX_MONEY])
    plt.show()

def Gambler_Proble():
    '''
    Main Function to solve this problem
    :return: Nein
    '''

    iter = 0
    value[0] = 0
    value[MAX_MONEY] = 1

    while True:
        old_Value = value.copy()
        value_iteration()
        gap = np.sum(abs(np.array(old_Value) - np.array(value)))
        if gap < CONVERGENCE_LIMIT:
            print('value iteration convergence in ' + str(iter) + ' iter: '+ str(gap))
            calculate_optimal_policy()
            print('the optimal state value is ' + str(value))
            print('the optimal policy is ' + str(policy))
            break
        iter = iter + 1
        print('iteration ' + str(iter) + ' : ' + str(gap))

# a = get_next_state_probability_reward(20, 15)
# print(a)
# print(a[0, 1])
#
Gambler_Proble()
