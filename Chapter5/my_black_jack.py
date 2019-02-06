import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

ACTION_HIT = 0
ACTION_STAND = 1
DISCOUNT = 0.9

def get_card():
    '''
        get a new card
    :return: card value,
    '''
    raw_card_value = np.random.randint(1, 14)
    card_value = min(10, raw_card_value)
    return card_value

def play(player_policy, dealer_policy):
    '''
    :param player_policy: typical player for this task
    :param dealer_policy: dealer for this task
    :return: trajectory([state-action pairs]), reward
    '''

    # =================initialize begin===================

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
    dealer_sum = 0
    dealer_card1_value = dealer_card1
    dealer_card2_value = dealer_card2
    if 1 in (dealer_card1, dealer_card2):
        dealer_ace_use = True
        if 1 == dealer_card1:
            dealer_card1_value = 11
        if 1 == dealer_card2 and 1 != dealer_card1:
            dealer_card2_value = 11
    dealer_sum = dealer_card1_value + dealer_card2_value

    # =================init end=================

    # game start!
    state_trajectory = []

    # player's turn
    while True:
        assert player_sum <= 22
        action = player_policy(player_sum)
        status = (player_ace_use, player_sum, dealer_card1)
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

def player_policy_cur(player_sum):
    if player_sum > 19:
        return ACTION_STAND
    else:
        return ACTION_HIT

def dealer_policy_cur(dealer_sum):
    if dealer_sum > 17:
        return ACTION_STAND
    else:
        return ACTION_HIT

    for pac in trajetory:
        status, action = pac
        print('status: ' + str(status))
        print('action: ' + str(action))
        # print(a, b, c)
    print('reward: ' + str(reward))

def monte_carlo_first_visit(episodes):
    no_ace_status_value_sum = np.zeros([10, 10])
    ace_status_value_sum = np.zeros([10, 10])
    no_ace_status_times = np.ones([10, 10])
    ace_status_times = np.ones([10, 10])

    for i in range(episodes):
        print('-----------episode: ' + str(i))
        trajetory, reward = play(player_policy_cur, dealer_policy_cur)
        trajetory.reverse()

        for pac in trajetory:
            reward = DISCOUNT * reward
            status, action = pac
            ace, player_sum, dealer_card = status
            print(pac)
            if ace is True:
                ace_status_times[player_sum-12, dealer_card-1] += 1
                ace_status_value_sum[player_sum-12, dealer_card-1] += reward
            else:
                no_ace_status_times[player_sum - 12, dealer_card - 1] += 1
                no_ace_status_value_sum[player_sum - 12, dealer_card - 1] += reward
    return no_ace_status_value_sum / no_ace_status_times, ace_status_value_sum / ace_status_times


def figure_5_1():
    no_ace_status_value, ace_status_value = monte_carlo_first_visit(100000)
    fig = plt.figure()

    ax = Axes3D(fig)
    X = np.array([[j for j in range(10)] for i in range(10)])
    Y = X.transpose().copy()
    X += 12
    Y += 1
    ax.plot_surface(X, Y, no_ace_status_value , rstride=10, cstride=10)
    ax.plot_surface(X, Y, ace_status_value , rstride=10, cstride=10)
    plt.show()

figure_5_1()
