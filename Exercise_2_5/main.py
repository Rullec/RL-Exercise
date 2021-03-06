import numpy as np
import matplotlib.pyplot as plt

class Paras:
    def __init__(self, _armNumber, _banditNum, _epsilon, _epoches, _stepsize, _ucb_para_c, _banditType):
        # bandit arms' number
        self.armNumber = _armNumber

        # bandits machine number
        self.banditNumber = _banditNum

        # greedy percent
        self.epsilon = _epsilon

        # running epoches
        self.epochNumber = _epoches

        # stepsize for weighted average
        self.stepsize = _stepsize

        # ucb parameter c
        self.c = _ucb_para_c

        # bandit problem algorithm type
        self.banditType = _banditType

class BanditMachine:
    """
    Including different tpyes of bandit algorithm

    """
    def __init__(self, _banditType, _armNumber, _banditNumber, _epsilon, _stepsize, _ucb_c, _if_optimisitic_initial_value):

        # bandit problem type
        self.banditType = _banditType

        # bandit number
        self.banditNumber = _banditNumber

        # arm number
        self.armNumber = _armNumber

        # epsilon greedy percent
        self.epsilon = _epsilon

        # stepsize for weighted average
        self.stepsize = _stepsize

        # ucb parameter c
        self.c = _ucb_c

        # estimated reward
        if _if_optimisitic_initial_value is False:
            self.q = np.zeros((self.banditNumber, self.armNumber))
        else:
            self.q = 5 * np.ones((self.banditNumber, self.armNumber))

        # sampletime - for sample_average
        self.sampleTime = np.zeros((self.banditNumber, self.armNumber))

        # current epoch
        self.t = 1

    def decide(self):
        A = []
        if ("sample_average" == self.banditType) or ("weighted_average" == self.banditType):
            #A = [np.argmax(row) if 0 == np.random.binomial(1,self.epsilon) else int(np.random.uniform(0,self.armNumber)) for row in self.q[:,]]
            A = [np.argmax(row) if np.random.binomial(1, self.epsilon) == 0 else int(np.random.uniform(0, self.armNumber)) for
                 row in self.q[:, ]]

        if ("ucb" == self.banditType):
            A = [np.argmax(row + self.c * np.sqrt(np.log(self.t ) / (n))) if np.array(n).size == np.array(np.array(n).nonzero()).size else np.where(n==0)[0][0] for row, n in zip(self.q[:,], self.sampleTime[:, ])]


        self.sampleTime = np.reshape(
            [self.sampleTime[i,] + BanditMachine.onehot(self.armNumber, value) for i, value in enumerate(A)],
            (self.banditNumber, self.armNumber))

        self.t = self.t + 1
        return A

    def update(self, reward, action):
        if "sample_average" == self.banditType:
            self.q = np.reshape([row + 1/self.sampleTime[i, action[i]]*(reward[i] - row[action[i]]) * BanditMachine.onehot(self.armNumber, action[i]) for i, row in enumerate(self.q[:,])], (self.banditNumber,self.armNumber))

        if "weighted_average" == self.banditType:
            # 对于weighted_average这个方法来说，如果一个动作不变动，那Q值需要改变吗?
            self.q = np.reshape([ row + self.stepsize * (reward[i] - row[action[i]]) * BanditMachine.onehot(self.armNumber, action[i]) for i, row in enumerate(self.q[:, ]) ], (self.banditNumber, self.armNumber))

        if "ucb" == self.banditType:
            print('ucb update!')
            self.q = np.reshape(
                [row + self.stepsize * (reward[i] - row[action[i]]) * BanditMachine.onehot(self.armNumber, action[i])
                 for i, row in enumerate(self.q[:, ])], (self.banditNumber, self.armNumber))
    @staticmethod
    def onehot(length, id):
        oh = np.zeros(length)
        oh[id] = 1
        return oh

class Testbed:
    """

    """
    def __init__(self, _armNumber, _questionNumber):
        self.armNumber = _armNumber
        self.questionNumber = _questionNumber

        self.Q =np.zeros((self.questionNumber, self.armNumber))

    def feedback(self, action):
        reward = [np.random.normal(self.Q[i, value], 1) for i, value in enumerate(action)]
        if_optimal = [np.argmax(row) == action[i] for i, row in enumerate(self.Q[:,])]
        assert self.questionNumber == len(reward)
        assert self.questionNumber == len(if_optimal)
        return reward, if_optimal

def main():

    parameter_sample_average = Paras(10, 2000, 0.1, 1000, None, None, 'sample_average')

    parameter_weighted_average = Paras(10, 2000, 0.1, 1000, 0.1, None,  'weighted_average')

    parameter_ucb =  Paras(10, 2000, 0.1, 1000, 0.1, 1,  'ucb')

    bandit1 = BanditMachine(parameter_sample_average.banditType, parameter_sample_average.armNumber, parameter_sample_average.banditNumber,
                            parameter_sample_average.epsilon, parameter_sample_average.stepsize, None, False)

    bandit2 = BanditMachine(parameter_weighted_average.banditType, parameter_weighted_average.armNumber, parameter_weighted_average.banditNumber,
                            parameter_weighted_average.epsilon, parameter_weighted_average.stepsize, None, False)

    bandit3 = BanditMachine(parameter_weighted_average.banditType, parameter_weighted_average.armNumber,
                            parameter_weighted_average.banditNumber,
                            parameter_weighted_average.epsilon, parameter_weighted_average.stepsize, None, True)

    bandit4 = BanditMachine(parameter_ucb.banditType, parameter_ucb.armNumber, parameter_ucb.banditNumber,
                            parameter_ucb.epsilon, parameter_ucb.stepsize, parameter_ucb.c, False)
    print(parameter_ucb.c)
    testbed = Testbed(parameter_sample_average.armNumber, parameter_sample_average.banditNumber)

    # average rewards
    ARs_sample_average = []    # average reward
    POAs_sample_average = []   # percent of Optimal Actions
    ARs_weighted_average = []    # average reward
    POAs_weighted_average = []   # percent of Optimal Actions
    ARs_optimisitic_initial = []    # average reward
    POAs_optimisitic_initial = []   # percent of Optimal Actions
    ARs_ucb = []
    POAs_ucb = []

    for i in range(parameter_sample_average.epochNumber):

        # sample average bandit
        A = bandit1.decide()
        R, if_optimal = testbed.feedback(A)
        bandit1.update(R, A)
        ARs_sample_average.append(np.average(R))
        POAs_sample_average.append(np.average(if_optimal))
        print('[sample average] ' + str(i) + ' epoch : average reward = ' + str(np.average(A)) + ' POA : ' + str(np.average(if_optimal)))

        # weighted average bandit
        A = bandit2.decide()
        R, if_optimal = testbed.feedback(A)
        bandit2.update(R, A)
        ARs_weighted_average.append(np.average(R))
        POAs_weighted_average.append(np.average(if_optimal))
        print('[weighted average] ' + str(i) + ' epoch : average reward = ' + str(np.average(A)) + ' POA : ' + str(np.average(if_optimal)))

        # weighted average bandit with optimistic initial value
        A = bandit3.decide()
        R, if_optimal = testbed.feedback(A)
        bandit3.update(R, A)
        ARs_optimisitic_initial.append(np.average(R))
        POAs_optimisitic_initial.append(np.average(if_optimal))
        print('[optimisitic initial] ' + str(i) + ' epoch : average reward = ' + str(np.average(A)) + ' POA : ' + str(np.average(if_optimal)))

        # ucb
        A = bandit4.decide()
        R, if_optimal = testbed.feedback(A)
        bandit4.update(R, A)
        ARs_ucb.append(np.average(R))
        POAs_ucb.append(np.average(if_optimal))
        print('[ucb] ' + str(i) + ' epoch : average reward = ' + str(np.average(A)) + ' POA : ' + str(np.average(if_optimal)))

        ## walk for a short distance(normal distribution)
        testbed.Q += np.random.normal(0, 0.01, (parameter_sample_average.banditNumber, parameter_sample_average.armNumber))


    plt.figure(0)
    plt.plot(ARs_sample_average, label='sample average')
    plt.plot(ARs_weighted_average, label='weighted average')
    plt.plot(ARs_optimisitic_initial, label='optimisitic initial')
    plt.plot(ARs_ucb, label='ucb')
    plt.title('average reward')
    plt.legend(loc='best')

    plt.figure(1)
    plt.plot(POAs_sample_average, label = 'sample average')
    plt.plot(POAs_weighted_average, label='weighted average')
    plt.plot(POAs_optimisitic_initial, label='optimisitic initial')
    plt.plot(POAs_ucb, label='ucb')
    plt.title('% optimal actions')
    plt.legend(loc='best')

    plt.show()

main()