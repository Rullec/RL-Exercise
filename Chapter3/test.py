'''
xudongfeng1996@gmail.comn

Figure 3.2 and 3.5's solution for Sutton's RL book draft
Figure 3.2: calculate state-value function for given random policy
Figure 3.5: calculation
'''

import numpy as np

SQUARE_EDGE = 5
POINT_A = [0, 1]
POINT_A_PRIME = [4,1]
POINT_B = [0,3]
POINT_B_PRIME = [2, 3]
DISCOUNT = 0.9
ACTION_SET = [np.array([0,1]), np.array([0,-1]),np.array([1,0]),np.array([-1, 0])]
CONSTANT_MOVE = 0

Value = np.zeros((SQUARE_EDGE, SQUARE_EDGE))

def go(now_pos, action):
    if POINT_A == now_pos:
        return POINT_A_PRIME, 10 + CONSTANT_MOVE
    if POINT_B == now_pos:
        return POINT_B_PRIME, 5 + CONSTANT_MOVE

    next_pos = now_pos + action
    reward = CONSTANT_MOVE
    if np.max(next_pos) >=SQUARE_EDGE or np.min(next_pos) <= -1:
        next_pos = now_pos
        reward = -1 + CONSTANT_MOVE

    return next_pos, reward

def Figure_3_2():
    iteration = 0
    while True:
        GAP = -1
        for i in range(SQUARE_EDGE):
            for j in range(SQUARE_EDGE):
                now_pos = [i, j]
                now_value = 0
                for action in ACTION_SET:
                    next_pos, reward = go(now_pos, action)
                    now_value = now_value + 0.25 * (reward + DISCOUNT * Value[next_pos[0], next_pos[1]])
                GAP = max(GAP, np.abs(now_value - Value[now_pos[0],now_pos[1]]))
                Value[now_pos[0],now_pos[1] ] = now_value
        iteration = iteration + 1
        print('iteration ' + str(iteration)  + ': ' + str(GAP))
        if GAP< 10e-12:
            break
    print('state-value function for random policy: \n' + str(Value))

def Figure_3_5():
    iteration = 0
    while True:
        GAP = -1
        for i in range(SQUARE_EDGE):
            for j in range(SQUARE_EDGE):
                now_pos = [i, j]
                values = []
                for action in ACTION_SET:
                    next_pos, reward = go(now_pos, action)
                    # now_value = now_value + 0.25 * (reward + DISCOUNT * Value[next_pos[0], next_pos[1]])
                    values.append(reward + DISCOUNT * Value[next_pos[0], next_pos[1]])
                now_value = np.max(values)
                GAP = max(GAP, np.abs(now_value - Value[now_pos[0],now_pos[1]]))
                Value[now_pos[0],now_pos[1] ] = now_value
        iteration = iteration + 1
        print('iteration ' + str(iteration) + ': ' + str(GAP))
        if GAP< 10e-10:
            break
    print('final optimal solutions: \n' + str(Value))


Figure_3_2()
Figure_3_5()