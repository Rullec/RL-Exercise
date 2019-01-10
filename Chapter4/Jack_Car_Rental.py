import numpy as np
from math import e, factorial

MAX_CAR_NUM = 20                # max capacity for rental point A and B
EARN_MONEY_RENT_PER_CAR = 10    # if you rent a car successfully, you can earn money for 10 euros
COST_MONEY_MOVE_PER_CAR = 2     # if you move a car, it costs 2 euros
MOVE_LIMIT = 5                  # the max num of moving cars in one night
DISCOUNT = 0.9
FLOAT_MIN = float("-inf")
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
        state = int(max(min(state, MAX_CAR_NUM), 0))
    elif list == type(state):
        assert 2==len(state)
        for i in range(0, 2):
            state[i] = int(max(min(state[i], MAX_CAR_NUM), 0))
    return state

def get_singel_point_next_state_reward_probability(rent_lambda, back_lambda, num_cars_after_night):
    '''
    :param rent_lambda: rental possion distribution parameter in this point
    :param back_lambda: back possion distribution parameter in this point
    :param now_state:   the number of cars AFTER night
    :param move_reward: the cost of moving cars over last night
    :return:     the indices of this matrix is 'next state', corrosponded [reward, probability].shape = (MAX_CAR_NUM + 1, 2),
                this reward only include the money earned by rental cars, do not contain the cost of moving cars overngiht
    '''
    assert int == type(num_cars_after_night)
    assert  0 <= num_cars_after_night <= MAX_CAR_NUM

    rent_probability_lambda = []
    back_probability_lambda = []
    if 3 == rent_lambda:
        rent_probability_lambda = probability_lambda_equal_3
    if 4 == rent_lambda:
        rent_probability_lambda = probability_lambda_equal_4
    if 2 == rent_lambda:
        rent_probability_lambda = probability_lambda_equal_2

    if 2 == back_lambda:
        back_probability_lambda = probability_lambda_equal_2
    if 3 == rent_lambda:
        back_probability_lambda = probability_lambda_equal_3
    if 4 == back_lambda:
        back_probability_lambda = probability_lambda_equal_4

    next_state_reward_probability = np.zeros((MAX_CAR_NUM + 1, 2))

    # rent cars in this point
    rent_probability_reward = np.zeros((num_cars_after_night+1, 2))

    for i in range(num_cars_after_night + 1):
        if i != num_cars_after_night:
            rent_probability_reward[i, 0] = rent_probability_lambda[i]
        else:
            rent_probability_reward[i, 0] = 1 - sum(rent_probability_reward[:num_cars_after_night, 0])
        rent_probability_reward[i, 1] = i * EARN_MONEY_RENT_PER_CAR
    # print('for rental cars, pro and reward matrix is : \n' + str(rent_probability_reward))
    # print('rent pro reward: ' + str(rent_probability_reward))
    # back cars in this point
    back_probability = np.array(back_probability_lambda)
    back_probability[-1] = 1 - np.sum(back_probability[0 : MAX_CAR_NUM])
    # print('for back cars, pro is \n' + str(back_probability))

    # combine different rent and back diff, shape a list which contains next_state(cars number), reward, probability
    sum_test = 0
    for rent_num in range(rent_probability_reward.shape[0]):
        for back_num in range(back_probability.shape[0]):
            # print('rent ' + str(rent_num) +' back ' + str(back_num))
            A_next_state = regular_state_num(num_cars_after_night - rent_num + back_num)
            transmission_pro = rent_probability_reward[rent_num, 0] * back_probability[back_num]
            single_reward = rent_probability_reward[rent_num, 1]
            sum_test = sum_test + transmission_pro * single_reward

            cur_state_reward = (next_state_reward_probability[A_next_state, 1] * next_state_reward_probability[A_next_state, 0] + transmission_pro * single_reward) / (transmission_pro + next_state_reward_probability[A_next_state, 1])

            next_state_reward_probability[A_next_state, 0] = cur_state_reward
            next_state_reward_probability[A_next_state, 1] = next_state_reward_probability[A_next_state, 1] + transmission_pro
    # print(next_state_reward_probability[:,1])

    assert (MAX_CAR_NUM + 1, 2) == next_state_reward_probability.shape
    assert abs(1 - np.sum(next_state_reward_probability[:,1])) < 1e-10
    assert abs(sum_test - np.sum(next_state_reward_probability[:,0] * next_state_reward_probability[:,1])) < 10e-10

    return next_state_reward_probability

def get_next_state_and_reward_pairs(now_state, action):
    '''
    :param now_state: a pair of numbers which range in [0, 20], represent the cars' number after that day
    :param action: a number from -5 to 5, the number of cars moved from B to A overnight
    :return: next_state_pro, next_state_reward
    '''
    assert abs(action) <= MOVE_LIMIT
    assert 0 <= now_state[0] <= MAX_CAR_NUM
    assert 0 <= now_state[1] <= MAX_CAR_NUM

    # print('get_next_state_and_reward for state: ' + str(now_state) + ', action: ' + str(action))
    move_reward = -1 * COST_MONEY_MOVE_PER_CAR * abs(action)
    now_state = regular_state_num([now_state[0] + action, now_state[1] - action])
    # print('after night, state changed to: ' + str(now_state))

    # decide point A and B 's rental cars' reward and corresponding probability
    A_state_reward_probability = get_singel_point_next_state_reward_probability(3, 3, now_state[0])
    B_state_reward_probability = get_singel_point_next_state_reward_probability(4, 2, now_state[1])

    next_state_probability = np.zeros((MAX_CAR_NUM+1, MAX_CAR_NUM+1))
    next_state_reward = np.zeros((MAX_CAR_NUM+1, MAX_CAR_NUM+1))

    for a in range(MAX_CAR_NUM+1):
        for b in range(MAX_CAR_NUM+1):
            next_state_probability[a, b] = A_state_reward_probability[a, 1] * B_state_reward_probability[b, 1]
            next_state_reward[a, b] = A_state_reward_probability[a, 0] + B_state_reward_probability[b, 0] + move_reward

    assert abs(1 - np.sum(next_state_probability)) < 1e-10
    return next_state_probability, next_state_reward

def calculate_value_function(cur_state):
    assert 2 == len(cur_state)
    assert 0 <= cur_state[0] <= MAX_CAR_NUM
    assert 0 <= cur_state[1] <= MAX_CAR_NUM

    next_state_probability, next_state_reward = get_next_state_and_reward_pairs(cur_state, Policy[cur_state[0], cur_state[1]])
    value_goal = 0
    for a in range(MAX_CAR_NUM + 1):
        for b in range(MAX_CAR_NUM + 1):
            value_goal = value_goal + next_state_probability[a,b] * (next_state_reward[a,b] + DISCOUNT * Value[a,b])
    # print(value_goal)

    return value_goal

def policy_evaluation():
    print('policy evaluation...')
    iter = 0
    while True:
        delta = 0
        old_Value = Value.copy()
        for i in range(MAX_CAR_NUM + 1):
            for j in range(MAX_CAR_NUM + 1):
                Value[i, j] = calculate_value_function([i, j])
                delta = max(abs(old_Value[i,j] - Value[i, j]), delta)
        iter = iter + 1
        print('policy evaluation, iter ' + str(iter))
        print('delta is ' + str(delta))
        if delta < 1e-3:
            # print(Value)
            print('policy evaluation done!')
            break

def find_optimal_action(now_state):
    assert 2 == len(now_state)
    assert 0 <= now_state[0] <= MAX_CAR_NUM
    assert 0 <= now_state[1] <= MAX_CAR_NUM

    action_value_list = []
    for action in range(-1 * MOVE_LIMIT, 1 + MOVE_LIMIT):
        if (now_state[1] - action <0) or (now_state[0] + action < 0):
            action_value_list.append(FLOAT_MIN)
            continue
        pro, reward = get_next_state_and_reward_pairs(now_state, action)
        action_value = 0
        for i in range(MAX_CAR_NUM + 1):
            for j in range(MAX_CAR_NUM + 1):
                action_value = action_value + pro[i, j] * (reward[i, j] + DISCOUNT * Value[i, j])
        action_value_list.append(action_value)
    optimal_action =  np.argmax(action_value_list) - MOVE_LIMIT

    assert  abs(optimal_action) <= MOVE_LIMIT
    return optimal_action

def policy_improvement():
    print('policy improvement...')
    changed_policy = 0
    for i in range(MAX_CAR_NUM + 1):
        for j in range(MAX_CAR_NUM + 1):
            old_action = Policy[i, j]
            Policy[i, j] = find_optimal_action([i, j])
            if old_action != Policy[i, j]:
                changed_policy = changed_policy + 1
                print(str([i, j]) + 'state policy chaned!')
    if 0 == changed_policy:
        print('optimal policy get!')
        return OPTIMAL
    else:
        print(str(changed_policy) + ' policies changed!')
        return UNOPTIMAL

def jack_car_rental():
    iter = 0
    while True:

        # policy evaluation
        policy_evaluation()

        # policy improvement
        if(OPTIMAL == policy_improvement()):
            print('get optimal policy, policy is \n' + str(Policy))

            break
        print(Policy)
        iter = iter + 1
        print('iter ' + str(iter) + ':')


jack_car_rental()

# a, b = get_next_state_and_reward_pairs([7,12], -3)

# policy_evaluation()
#
# print(Value)
#
# policy_improvement()
#
# print(Policy)
# policy_improvement()

# a = get_singel_point_next_state_reward_probability(3, 3, 1)
# print(a)
# a, b= get_next_state_and_reward_pairs([0,0],1)
# print(a)
# print(b)