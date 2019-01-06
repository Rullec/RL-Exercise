import numpy as np
import matplotlib.pyplot as plt

class Paras:
    def __init__(self, _armNumber, _banditNum, _epsilon, _epoches, _banditType):
        # bandit arms' number
        self.armNumber = _armNumber

        # bandits machine number
        self.banditNumber = _banditNum

        # greedy percent
        self.epsilon = _epsilon

        # running epoches
        self.epochNumber = _epoches

        # bandit problem algorithm type
        self.banditType = _banditType

class BanditMachine:
    """
    Including different tpyes of bandit algorithm

    """
    def __init__(self, _banditType, _armNumber, _banditNumber, _epsilon):

        # bandit problem type
        self.banditType = _banditType

        # bandit number
        self.banditNumber = _banditNumber

        # arm number
        self.armNumber = _armNumber

        # epsilon greedy percent
        self.epsilon = _epsilon

        # estimated reward
        self.q = np.zeros((self.banditNumber, self.armNumber))

        # sampletime - for sample_average
        self.sampleTime = np.zeros((self.banditNumber, self.armNumber))

    def decide(self):
        A = []
        if "sample_average" == self.banditType:
            A = [np.argmax(row) if 0 == np.random.binomial(1,self.epsilon) else int(np.random.uniform(0,self.armNumber)) for row in self.q[:,]]
            self.sampleTime = np.reshape([self.sampleTime[i,] + BanditMachine.onehot(self.armNumber,value) for i, value in enumerate(A)], (self.banditNumber, self.armNumber))

        #print(len(A[:,]))
        # print(len(A[:,]) == self.banditNumber)#, 'wrong in decide function: len %d != %d'.format(len(A[:,]), self.banditNumber)
        return A

    def update(self, reward, action):
        if "sample_average" == self.banditType:
            self.q = np.reshape([row + 1/self.sampleTime[i, action[i]]*(reward[i] - row[action[i]]) * BanditMachine.onehot(self.armNumber, action[i]) for i, row in enumerate(self.q[:,])], (self.banditNumber,self.armNumber))

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

    paras = Paras(10, 20, 0.1, 10000, 'sample_average')

    bandit1 = BanditMachine(paras.banditType, paras.armNumber, paras.banditNumber,
                            paras.epsilon)

    testbed = Testbed(paras.armNumber, paras.banditNumber)

    # average rewards
    ARs = []    # average reward
    POAs = []   # percent of Optimal Actions
    for i in range(paras.epochNumber):

        # bandit decide
        A = bandit1.decide()

        # testbed give reward, judge POA(), judge AR
        R, if_optimal = testbed.feedback(A)
        # bandit get reward and
        bandit1.update(R, A)

        ## walk for a short distance(normal distribution)
        testbed.Q += np.random.normal(0, 0.01, (paras.banditNumber, paras.armNumber))
        ARs.append(np.average(R))
        POAs.append(np.average(if_optimal))
        print(str(i) + ' epoch : average reward = ' + str(np.average(A)) + ' POA : ' + str(np.average(if_optimal)))

    plt.figure(0)
    plt.plot(ARs, label='sample average')
    plt.title('average reward')
    plt.legend(loc='best')

    plt.figure(1)
    plt.plot(POAs, label = 'sample average')
    plt.title('% optimal actions')
    plt.legend(loc='best')

    plt.show()

main()