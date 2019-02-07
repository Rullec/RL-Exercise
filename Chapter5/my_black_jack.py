import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

ACTION_HIT = 0
ACTION_STAND = 1
DISCOUNT = 0.9
PLAYER_POLICY = None

def get_card():
    '''
        get a new card
    :return: card value,
    '''
    raw_card_value = np.random.randint(1, 14)
    card_value = min(10, raw_card_value)
    return card_value

def play(player_policy, dealer_policy, init_status = None, init_action = None):
    '''
    :param player_policy: typical player for this task
    :param dealer_policy: dealer for this task
    :param init_status: if you need to select init status by yourself, you can pass in this para
    and the format is init_status = (player_sum, player_ace_use, dealer_card1)
    :param init_action: ACTION_HIT or ACTION_STAND for the player
    :return: trajectory([state-action pairs]), reward
    '''

    # =================initialize begin===================
    player_ace_use = False
    player_sum = 0
    dealer_ace_use = False
    dealer_card1 = get_card()
    dealer_card2 = get_card()
    dealer_sum = 0

    if init_action is None and init_status is None:
        # init player
        player_ace_use = False
        player_sum = 0
        while player_sum < 12:
            card = get_card()
            if 1 == card and player_ace_use is False:
                player_ace_use = True
                card = 11
            player_sum += card

        if player_sum > 21:
            assert 22 == player_sum
            assert player_ace_use is True
            player_sum -= 10
            player_ace_use = False

        # init dealer
        dealer_ace_use = False
        dealer_card1 = get_card()
        dealer_card2 = get_card()
        dealer_card1_value = dealer_card1
        dealer_card2_value = dealer_card2
        if 1 in (dealer_card1, dealer_card2):
            dealer_ace_use = True
            if 1 == dealer_card1:
                dealer_card1_value = 11
            if 1 == dealer_card2 and 1 != dealer_card1:
                dealer_card2_value = 11
        dealer_sum = dealer_card1_value + dealer_card2_value

    elif init_status is not None and init_action is not None:
        player_sum, player_ace_use, dealer_card1 = init_status

        dealer_ace_use = False
        dealer_card2 = get_card()
        dealer_card1_value = dealer_card1
        dealer_card2_value = dealer_card2
        if 1 == dealer_card1:
            dealer_ace_use = True
            dealer_card1_value = 11
        if 1 == dealer_card2 and dealer_ace_use is False:
            dealer_ace_use = True
            dealer_card2_value = 11
        dealer_sum = dealer_card2_value + dealer_card1_value

    else:
        assert False
    # =================init end=================

    # game start!
    state_trajectory = []

    # player's turn
    while True:
        assert player_sum < 22
        status = (player_sum, player_ace_use, dealer_card1)
        if 0 == len(state_trajectory) and init_action is not None:
            action = init_action
        else:
            action = player_policy(status)

        state_trajectory.append([status, action])
        if ACTION_STAND == action:
            break

        # 在这里的card如果面是1，一定不能被当成ace，不然的话12 + 11 = 23就爆炸
        card = get_card()
        player_sum += card
        if player_sum > 21 and player_ace_use is True:
            player_ace_use = False
            player_sum -= 10

        # boom up, reward = -1
        if player_sum > 21:
            return state_trajectory, -1

    # dealer's turn
    # dealer's policy is fixed
    while True:
        action = dealer_policy(dealer_sum)
        if ACTION_STAND == action or dealer_sum > 21:
            break

        # HIT
        card = get_card()
        dealer_sum += card
        if dealer_sum > 21 and dealer_ace_use is True:
            dealer_sum -= 10
            dealer_ace_use = False

    assert player_sum <= 22

    if dealer_sum > 21 or dealer_sum < player_sum:
        # boom up, player win
        return state_trajectory, 1
    elif dealer_sum > player_sum:
        return state_trajectory, -1
    elif dealer_sum == player_sum:
        return state_trajectory, 0

def player_policy_fix(status):
    global PLAYER_POLICY
    assert PLAYER_POLICY is not None
    assert len(PLAYER_POLICY) == 22

    player_sum, _, _ = status

    return PLAYER_POLICY[player_sum]

def player_policy_changing(status):
    '''

    :param status: (player_sum, player_ace_use, dealer_card1)
    :return:
    '''

    global PLAYER_POLICY
    assert (2, 10, 10) == PLAYER_POLICY.shape
    player_sum, player_ace_use, dealer_card1 = status
    return PLAYER_POLICY[int(player_ace_use), player_sum-12, dealer_card1-1]

def dealer_policy(dealer_sum):
    if dealer_sum > 17:
        return ACTION_STAND
    else:
        return ACTION_HIT

def monte_carlo_first_visit_only_evaluation(episodes):
    global PLAYER_POLICY
    PLAYER_POLICY = np.array([ACTION_STAND if i > 19 else ACTION_HIT for i in range(22)])

    no_ace_status_value_sum = np.zeros([10, 10])
    ace_status_value_sum = np.zeros([10, 10])
    no_ace_status_times = np.ones([10, 10])
    ace_status_times = np.ones([10, 10])

    for i in range(episodes):
        if i % 10000 == 0:
            print('episode: ' + str(i))
        trajetory, reward = play(player_policy_fix, dealer_policy)
        trajetory.reverse()

        for pac in trajetory:
            reward = DISCOUNT * reward
            status, action = pac
            player_sum, ace,  dealer_card = status
            if ace is True:
                ace_status_times[player_sum-12, dealer_card-1] += 1
                ace_status_value_sum[player_sum-12, dealer_card-1] += reward
            else:
                no_ace_status_times[player_sum - 12, dealer_card - 1] += 1
                no_ace_status_value_sum[player_sum - 12, dealer_card - 1] += reward
    return no_ace_status_value_sum / no_ace_status_times, ace_status_value_sum / ace_status_times

def policy_action_random_generate():
    '''
    generate random status and value,
    :return: (player_sum, player_ace_use, dealer_card1), action
    '''
    player_ace_use = False if 1 == np.random.randint(1,3) else True
    player_sum = np.random.randint(12, 22)
    dealer_card1 = get_card()
    action = ACTION_HIT if 1 == np.random.randint(1,3) else ACTION_STAND
    return (player_sum, player_ace_use, dealer_card1), action


def monte_carlo_es(episode):
    '''
        Monte Carlo exploring starts
    :param episode:
    :return:
    '''
    # initialize
    '''
    0. status = player_ace_use * player_sum * dealer_card1 = 2 * 10 * 10 = 200
    1. player's policy, player[status]
    2. value function for the player, Q[status, 2]
    3. Returns value for the player, R[status, 2]
    4. Sample times for the player, Time[status, 2]
    '''
    global PLAYER_POLICY
    PLAYER_POLICY = np.zeros([2, 10, 10], dtype = int)
    value_func = np.zeros([2, 10, 10, 2], dtype = float)
    sum_value = np.zeros([2, 10, 10, 2], dtype = float)
    sample_time = np.ones([2, 10, 10, 2], dtype = int)

    # loop begin
    for i in range(episode):
        status, action = policy_action_random_generate()
        tra, reward = play(player_policy_changing, dealer_policy, status, action)
        print('----------------' + str(i))
        tra.reverse()
        for status, action in tra:
            reward = DISCOUNT * reward
            player_sum, player_ace_use, dealer_card1 = status
            player_ace_use = int(player_ace_use)
            sum_value[player_ace_use, player_sum - 12, dealer_card1 - 1, action] += reward
            sample_time[player_ace_use, player_sum-12, dealer_card1-1, action] += 1

        # update PLAYER_POLICY
        value_func = sum_value / sample_time
        PLAYER_POLICY = np.argmax(value_func, axis = 3)

    return np.max(value_func, axis = 3)

def figure_5_1():
    no_ace_status_value, ace_status_value = monte_carlo_first_visit_only_evaluation(100000)
    fig = plt.figure()

    ax = Axes3D(fig)
    X = np.array([[j for j in range(10)] for i in range(10)])
    Y = X.transpose().copy()
    X += 12
    Y += 1
    ax.plot_surface(X, Y, no_ace_status_value , rstride=10, cstride=10)
    ax.plot_surface(X, Y, ace_status_value , rstride=10, cstride=10)
    plt.show()

def figure_5_2():
    value = monte_carlo_es(100000)  # [2, 10, 10]
    print(value.shape)
    no_ace_value = value[0, :, :]
    ace_value = value[1, :, :]
    print(no_ace_value.shape)
    print(ace_value.shape)

    fig = plt.figure()

    ax = Axes3D(fig)
    X = np.array([[j for j in range(10)] for i in range(10)])
    Y = X.transpose().copy()
    X += 12
    Y += 1
    ax.plot_surface(X, Y, no_ace_value , rstride=10, cstride=10)
    ax.plot_surface(X, Y, ace_value , rstride=10, cstride=10)
    plt.show()

    # display optimal policy
    no_ace_policy = PLAYER_POLICY[0, :, :]
    ace_policy = PLAYER_POLICY[1, :, :]
    print(no_ace_policy)
    # print(no_ace_policy.shape)
    # ax.plot_surface(X, Y, no_ace_policy , rstride=10, cstride=10)
    # ax.plot_surface(X, Y, ace_policy , rstride=10, cstride=10)
    # plt.show()

# figure_5_1()

figure_5_2()