import random
import matplotlib.pyplot as plt

# Global variable definition
epsilon = 0.1   # e-greedy strategy
arm_num = 10      # k armed question

# Learner definition
class Learner2_constant_step_size:
    def __init__(self):
        self.weight = [0] * arm_num
        self.action = -1
        self.step = 0.1
        self.times = [0] * arm_num
        self.avg_reward = []
        self.epoch = 0

    def update(self, curr_action, curr_value):
        assert curr_action <= arm_num
        assert curr_action >= 1
        self.weight[curr_action - 1] = self.weight[curr_action - 1] + self.step * (curr_value - self.weight[curr_action - 1])

        if 0 != self.epoch:
            self.avg_reward.append(self.avg_reward[-1]*(self.epoch-1))
        else:
            self.avg_reward.append(curr_value)
        self.epoch = self.epoch + 1

    def decide(self):
        if random.random() < epsilon:   # explore
            self.action = random.randint(1, arm_num)
        else:                           # exploration
            self.action = self.weight.index(max(self.weight)) + 1
        assert self.action <= arm_num
        assert self.action >= 1

        self.times[self.action - 1] = self.times[self.action - 1 ] + 1

        return self.action

class Learner1_sample_average:
    def __init__(self):
        self.weight = [0] * arm_num
        self.action = -1
        self.times = [0] * arm_num

    def update(self, curr_action, curr_value):
        assert curr_action <= arm_num
        assert curr_action >= 1
        self.weight[curr_action - 1] = self.weight[curr_action - 1] + (curr_value - self.weight[curr_action - 1]) / self.times[curr_action-1]

    def decide(self):
        if random.random() < epsilon:   # explore
            self.action = random.randint(1, arm_num)
        else:                           # exploration
            self.action = self.weight.index(max(self.weight)) + 1
        assert self.action <= arm_num
        assert self.action >= 1

        self.times[self.action - 1] = self.times[self.action - 1 ] + 1

        return self.action

class Testbed:
    def __init__(self):
        self.realQ = [0] * arm_num
        for i in range(arm_num):
            self.realQ[i] = random.random() * 10

    def feedback(self, action):
        return self.realQ[action - 1]

def main():
    testbed = Testbed()
    L1 = Learner1_sample_average()
    L2 = Learner2_constant_step_size()
    for i in range(10000):
        action_L2 = L2.decide()
        value_L2 = testbed.feedback(action_L2)
        L2.update(action_L2, value_L2)

        action_L1 = L1.decide()
        value_L1 = testbed.feedback(action_L1)
        L1.update(action_L1, value_L1)
        print("epoch " + str(i))

    print("L1 weight：" + str(L1.weight))
    print("L2 weight：" + str(L2.weight))
    print("test bed: " + str(testbed.realQ))

main()