import numpy as np
import tensorflow as tf
import gym
import traceback
from collections import deque
import os
from actor import Actor
from critic import Critic
import random
import time
import matplotlib.pyplot as plt

'''
    下面的代码将实现一个DDPG算法(Deep Deterministic Policy Gradient)的agent。
    这种方法和以前的实现不同: 他可以用于连续行为空间的控制(continuous action space)
    详细的网络结构、公式推导、训练方法，请看"DDPG.md"

    所谓确定性策略(deterministic category)，是和连续行为空间(continuous action space)绑定的;
    相对的(?)随机性策略(stochastic category), 就可以解决离散行为空间(discrete action space)绑定的。

    1. 网络结构 网络的输入输出
        DDPG算法中，一共有2对actor-critic网络。一对是用于控制的actor-critic网络，另一对是用于训练critic的target actor-critic网络。共计4个网络，分别为:
        - 主actor网络μ(s), 参数为theta^μ; 输入的是当前state，输出是action一个实数值: 
            在确定性(deterministic)策略中，这个action是一个实数值
            这一点和离散行为空间下不同: 离散行为空间中，输入一个state, 输出一个action的概率分布。
            输入不变，输出变化了。
        - 主critic网络Q(s, a), 参数为theta^Q; 输入是当前state和当前actoin: 输出是Q(s,a)一个实数值。
            在确定性策略中，critic网络的输入是一个state和一个action，输出的是一个Q(s,a)的实数值
            在随机性策略(离散行为空间)中，critic网络的输入只有一个state, 输出是Q(s,a) = [Q(s,a1), Q(s,a2), ... , Q(s,an)]一个向量
            输入输出都变化了。
        - target actor网络, 结构和主actor相同，不参与决策，只起辅助critic训练用
        - target critic网络，结构和主critic相同，不参与决策，只起辅助critic训练用
        关于"target网络"的思想，请参看含target的DQN网络的训练过程。
        
    2. 网络的训练方式:[详细信息请查看笔记]
        - 主网络actor训练: 策略梯度; 求策略梯度时，必然要涉及当前Q(s,a)的计算。
            最简单的policy gradient中，我们使用MC法来估计Q(s,a)，即Q(s,a) = \sum \gamma * reward
            而引入actor critic后，其实就是改MC法为TD法来估计Q(s,a), 即用一个网络来拟合Q(s, a)的值;
            所以在训练DDPG这一应用了actor-critic原则、使用TD法来估计Q(s,a)的值中的actor时
            就必然涉及对Q网络的传播: 在这个传播过程中我们手动fix critic网络，不让他移动。
            而把train 主critic网络的任务分到另一个部分
        - 主网络critic训练: 和传统的Q网络训练的方法几乎一样，唯一区别在于主网络的训练多需要一个q(s_, a_), 此时a_要从actor中来
    
    3. 一些澄清:
        由于我们使用TD来估计值，那么我们就不需要等episode跑完之后才能计算return进而估计梯度了。
        我们会在每一个transition结束时，训练一次critic,训练一次actor。每次训练都从buffer中抽一个minibatch
        每次运行完一个transition，都把这个transition存到buffer里面去。
'''
log_dir = "logs/"
log_name = "a2c.train.log"
log_path = log_dir + log_name

if False == os.path.exists(log_dir):
    os.makedirs(log_dir)
if os.path.exists(log_path):
    os.remove(log_path)

class DDPGAgent:
    def __init__(self):
        # init variables
        self.gamma = 0.99   # reward decay to return
        self.max_explore_iter = 20000
        self.max_len_buffer = 10000
        self.buffer = deque(maxlen=self.max_len_buffer)    # transition replay buffer
        self.lr_a = 0.0001   # actor 学习率
        self.lr_c = 0.001   # critic　学习率
        self.cur_epoch = 0  # 初始话epoch个数
        self.max_episode_step = 200 # 一个回合最多200步

        # create env, it must be continuous control problem
        self.env = self.create_env()

        # create replacement policy
        replacement = [
            dict(name='soft', tau = 0.001),
            dict(name='hard', rep_iter_actor= 600, rep_iter_critic=500)
        ][0]

        self.sess = tf.Session()
        

        # build network
        with tf.variable_scope("Actor"):
            self.actor = Actor(32, self.state_dims, self.action_dims, \
                replacement, self.action_low_bound, self.action_high_bound,\
                self.lr_a, self.max_explore_iter, self.sess
            )
        with tf.variable_scope("Critic"):
            self.critic = Critic(32, self.state_dims, self.action_dims,\
                self.lr_c, self.gamma, replacement, self.sess)
            
        writer = tf.summary.FileWriter("logs", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        
        return
    
    def create_env(self, env_name = "Pendulum-v0"):
        try:
            # create
            env = gym.make(env_name)
            env.reset()
            assert type(env.action_space) == gym.spaces.box.Box
            env._max_episode_steps = 200

            # init
            self.state_dims = env.observation_space.shape[0]
            # self.action_dims = env.action_space.n
            self.action_dims = 1
            self.action_high_bound = env.action_space.high
            self.action_low_bound = env.action_space.low
            return env

        except Exception as e:
            traceback.print_exc(e)
            print("[error] the name of env is %s" % env_name)

    def remember(self, state, action, reward, next_state):
        # 这次整个网络都用tensorflow写，要求所有传进来的都是numpy
        # print(type(action))
        # print(action.shape)
        assert action.shape == (1, self.action_dims)
        assert state.shape == (1, self.state_dims)
        assert next_state.shape == (1, self.state_dims)
        assert type(reward) == float

        self.buffer.append((state, action, reward, next_state))

        return 

    def replay(self, batch_size):
        '''
            采一个minibatch然后进行训练
        '''
        # print("replay")
        # 1. 先采样
        # t1 = time.time()
        sample_list = random.sample(self.buffer, batch_size)
        batch_state, batch_action, batch_reward, batch_next_state = [], [], [], []
        for state, action ,reward, next_state in sample_list:
            batch_state.append(state)
            batch_action.append(action)
            batch_reward.append(reward)
            batch_next_state.append(next_state)
        batch_state = np.reshape(np.array(batch_state), [batch_size, self.state_dims])
        batch_action = np.reshape(np.array(batch_action), [batch_size, self.action_dims])
        batch_reward = np.reshape(np.array(batch_reward), [batch_size])
        batch_next_state = np.reshape(np.array(batch_next_state), [batch_size, self.state_dims])

        assert batch_state.shape == (batch_size, self.state_dims)
        assert batch_action.shape == (batch_size, self.action_dims)
        assert batch_reward.shape == (batch_size, )        
        assert batch_next_state.shape == (batch_size, self.state_dims)


        # 开始训练critic
        # 1. 首先要把next_state都给到target actor,得到输出a_
        batch_next_action = self.actor.get_target_action(batch_next_state)
        assert batch_next_action.shape == (batch_size, self.action_dims)
        # 2. 把next_state和next_action送到target critic里，得到输出q'(s_,a_)
        batch_q_s_a_next = self.critic.get_target_q(batch_next_state, batch_next_action)
        assert batch_q_s_a_next.shape == (batch_size, )
        # 3. 然后把他们一起送到critic.learn里面去，进行一次训练，顺便计算一次dqda的梯度
        dqda = self.critic.train(batch_state, batch_action, batch_reward, batch_q_s_a_next)
        
        # print(dqda.shape)
        # print(dqda.transpose())
        assert dqda.shape == (batch_size, self.action_dims)
        
        # 开始训练actor
        self.actor.train(batch_state, dqda)

    def get_action(self, state):
        action = self.actor.get_action(state)
        assert type(action) == float
        return action

    def learn(self, batch_size = 64):
        state = self.env.reset()
        batch_reward = []
        for _ in range(self.max_episode_step):
            action = self.actor.get_action(np.reshape(state, [1, self.state_dims]))
            next_state, reward, done, _ = self.env.step(action)
            batch_reward.append(reward)

            # 记忆
            self.remember(np.reshape(state, [1, self.state_dims]), action, float(reward), np.reshape(next_state, [1, self.state_dims]))

            # 如果满了的话，训练
            if len(self.buffer) > batch_size:
                self.replay(batch_size)
            
            # 更新状态
            state = next_state

            # 如果停止了, 退出
            if done == True:
                break
        
        batch_length = len(batch_reward)
        batch_sum_reward = np.sum(batch_reward)
        
        return batch_length, batch_sum_reward

    def test(self):
        # print((self.action_low_bound, self.action_high_bound))
        state = self.env.reset()
        while True:
            self.env.render()
            action = self.actor.get_action(np.reshape(state, [1, self.state_dims]), test = True)
            # print(action)
            state, reward, done, _ = self.env.step(action)
            # print((action, reward))
            if done :
                break
        self.env.close()
    
    def get_epsilon(self):
        return self.actor.get_epsilon()

    def load(self, name):
        pass

    def save(self, name):
        pass

plt.ion()
if __name__ == "__main__":
    # print("SUCC")
    agent = DDPGAgent()
    epochs = 10000
    epoch_reward = []
    epoch_length = []
    print_gap = 1
    test_gap = 50
    for i in range(1, epochs):
        
        length, reward = agent.learn(batch_size=256)
        epoch_reward.append(reward)
        epoch_length.append(length)

        if i % print_gap == 0:
            print(" epoch %d: reward %.3f, length %.3f, epsilon %.3f" % (i, np.mean(epoch_reward[-print_gap:]), np.mean(epoch_length[-print_gap:]), agent.get_epsilon()))

            # epoch_length.clear()
            # epoch_reward.clear()
            
            plt.plot(epoch_reward)
            plt.pause(0.1)
            plt.cla()
            # plt.show()
            # plt.close()
            if reward > -700:
                agent.test()