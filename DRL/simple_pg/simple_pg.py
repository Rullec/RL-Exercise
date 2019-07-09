import tensorflow as tf
import numpy as np
import gym
import time
from gym import envs, spaces
import traceback

'''
    下面的代码实现了最简单的policy gradient，参考了https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
    中的代码，即spinup/examples/pg_math/1_simple_pg.py
    1. 算法的结构模型:
        策略梯度(policy gradient)是一种强化学习算法，能够解决控制问题。控制问题的核心目标是得到一个好的控制器(controller)
        这个controller的输入是当前state,输出是一个action，以此实现对角色的控制目的。
        什么是好的controller？用MDP的语言来说，就是如果角色服从controller所给出的action建议，那么这个角色的E[R(\tau)]就是很高的
        
        强化学习的基础是MDP，强化学习的核心目标是求解最优bellman方程。要求解bellman方程，就要对值函数(Q(s,a)以及V(s))
        进行估计。每一个值函数V(s)背后都有一个策略作支撑:只有给定策略时，我们才能说他的值函数是多少。
        估计值函数方法的方法分成两类:
            1. Monte Carlo采样
                V(s) = E[R(\tau) | s]
                Q(s, a) E[R(\tau) | s, a]
            2. TD法
                Q(s, a) = Q(s,a) + \alpha(r(s,a) + \gamma max_a{Q(s,a)} - Q(s,a))
        on policy算法使用Monte Carlo采样估计值函数: 例如PG, VPG等
        off policy算法使用TD法估计值函数: 例如DQN，DDPG

        在PG算法中，为了求解策略梯度，存在一个log \pi(\theta | a)的求和过程
        我们需要policy函数进行拟合和优化。在这里，选取神经网络作为policy函数的basis
        也就是说建立一个含有参数theta的policy网络,之后不断地对其中的参数theta进行优化
        争取让这个policy变得越来越好，指导获得的return越来越高。
        loss = -J(\theta) = -E[R(\tau)]
        
    2. 算法的执行过程:
        0. build一个policy网络，这个网络的输入是state，输出是action的概率分布:
            以CartPole-v0举例，这个问题有3个可能的action:向左、向右、原地不动，即
            policy网络的输入是当前的state_{4*1}，输出是一个1*3的向量[0.1, 0.7, 0.2]，且sum(向量)=1
            这个向量中的三个float，就是采取对应action的概率。
            网络由2层FC组成;网络的loss = J(\theta)，对J(\theta)求导得到的正好是策略梯度。
            关于loss的构建和策略梯度究竟是如何计算的，看"Part 3-1 Intro to Policy Optimization"第二部分
        
        1. while True:
            采样1个episode，得到所有的state-action对
            例如，在本次采样episode中，经过了179个s-a对(称之为transition)
            则全部送入网络，进行policy gradient的下降、训练
            返回采样步骤，不断训练。
        
        
'''

class PGAgent:

    def __init__(self):
        return

    def compute_reward(self, reward, gamma):
        '''
            add spinup - part 3 - Don’t Let the Past Distract You - reward to go
            add gamma
        '''
        assert type(reward) is list
        assert len(reward) > 0

        length = len(reward)
        return_ = np.zeros(length)
        temp = 0
        for i in range(length):
            temp = reward[-(i+1)] + gamma * temp
            return_[-(i+1)] = temp
        return return_.tolist()

    def create_env(self, env_name = "CartPole-v0"):
        #  MountainCar-v0, CartPole-v0
        try:
            self.env = gym.make(env_name)
            self.env.reset()
            self.action_dim = self.env.action_space.n
            self.state_dim = self.env.observation_space.shape[0]
        except Exception as e:
            traceback.print_exc(e)
            print("[error] the name of env is %s" % env_name)

    def build_network(self, learning_rate=0.01):
        assert self.action_dim > 0 and self.env is not None and self.state_dim >0 
        self.input_layer = tf.placeholder(shape = (None, self.state_dim), dtype = tf.float32, name="input_layer")

        self.mid_layer = tf.layers.dense(self.input_layer, units = 32, activation = tf.nn.relu)

        # 这一层输出的是各个action的概率
        self.output_layer = tf.layers.dense(self.mid_layer, units = self.action_dim, activation = None)

        # 这一层依照上面的概率进行采样
        self.action_layer = tf.squeeze(tf.random.categorical(logits = self.output_layer, num_samples = 1), axis = 1)    #shape = (?, 1)
        # self.action_layer = tf.random.categorical(logits = self.output_layer, num_samples = 1)    # shape = (?, )
        
        # 计算batchsize个轨迹(trajectory)的概率
        self.action_ph = tf.placeholder(dtype = tf.int32, shape = (None, ), name = "action_placeholder")
        self.return_ph = tf.placeholder(dtype = tf.float32, shape = (None, ), name = "return_placeholder")
        action_mask = tf.one_hot(self.action_ph, self.action_dim, name = "action_mask")
        log_trajectory_probs = tf.reduce_sum(action_mask * tf.nn.log_softmax(self.output_layer), axis=1)# 这些轨迹发生的概率的log是什么
        # tf.shape(log_trajectory_probs) = (batch_size, )

        # 计算loss: 需要乘以return
        self.loss = -tf.reduce_mean(self.return_ph * log_trajectory_probs)
        
        # train
        self.train = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.loss)

        # init
        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)


    def init(self, env_name = "CartPole-v0", batch_size = 5000, render = False):
        
        # variable init
        self.env_name = env_name
        self.batch_size = batch_size
        self.render_switch = render

        # create environment
        self.create_env(env_name)
        assert self.action_dim >0 and self.state_dim > 0

        # build 
        self.build_network()

    def train_one_step(self, batch_size, reward_to_go = False, gamma=0.99):
        # 先收集数据，跑batch size个episode的act
        assert isinstance(self.env, gym.Env) and self.init is not None
        batch_states = []
        batch_actions = []
        batch_returns = []
        
        cur_state = self.env.reset()
        episode_reward = []
        
        while True:
            # run the policy network to get an action
            # print(cur_state.shape)  
            cur_state = cur_state.reshape(1, self.state_dim)    # 变形，适应输入: 这里的input_layer要求(None, 4)，也就是说必须是(1, 4)形, (4, )是不够的
            action = self.sess.run(self.action_layer, feed_dict={self.input_layer : cur_state})[0]
            batch_states.append(cur_state.reshape(1, -1))
            
            # then take the action in env, get return and state feed back
            cur_state, reward, episode_finished, _ = self.env.step(action)

            batch_actions.append(action)
            episode_reward.append(reward)

            if episode_finished == True:
                
                # 我们最后要的是actions, returns(相同shape), states
                # 如果总共的state数已经攒够batch size了(例如5004)，就停止收集数据去训练
                # episode_return += 201
                batch_returns += self.compute_reward(episode_reward, gamma=gamma)

                # 当前episode终止
                episode_reward, episode_finished, cur_state = [], False, self.env.reset()
                if len(batch_states) > batch_size:
                    break
         
        # 开始训练，并且获得loss和return
        batch_actions = np.array(batch_actions)
        batch_states = np.array(batch_states).reshape(-1, self.state_dim)
        
        batch_returns = np.array(batch_returns)
        # print(batch_states.shape)
        # print(len(batch_states))
        # print(type(batch_actions))
        # print(batch_actions.shape)
        _, loss = self.sess.run([self.train, self.loss], feed_dict={
                                        self.action_ph: batch_actions,
                                        self.return_ph: batch_returns,
                                        self.input_layer : batch_states})

        
        return loss, np.mean(batch_returns)

    def test(self):
        state = self.env.reset()
        
        while True:
            self.env.render()
            action = self.sess.run(self.action_layer, feed_dict={
                self.input_layer: state.reshape(1, self.state_dim)
            })[0]
            state, reward, done,_ = self.env.step(action)
            if done == True:
                break

        return


if __name__ == "__main__":
    NN = PGAgent()
    NN.create_env()
    NN.build_network(learning_rate=0.1)
    epochs = 100

    for i in range(epochs):
        loss, batch_return = NN.train_one_step(batch_size = 2000)
        
        print("epoch %d: loss: %.3f, avg_return: %.3f" % (i, loss, batch_return))
        if i % 20 == 0:
            NN.test()
