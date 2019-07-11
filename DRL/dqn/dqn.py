import random
import gym
import numpy as np
import traceback
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os
import sys

'''
    基础知识请参看simple_pg.py
    下面的代码实现了DQN，详细推导请参看"Q-learning(DQN)"笔记。

    DQN(Deep Q Network)是一种off-policy强化学习算法，使用TD方法来对(action)值函数进行估计。
    1. 算法的结构模型:
        DQN算法，顾名思义，就是使用神经网络来拟合action value function。
        网络输入: state 网络输出: Q(s,a)
        只要获得这样一个网络，就可以 action = argmax_a Q(s,a)进行控制了。
        loss定义: 全盘照抄Tabular TD，loss是TD error的L2 norm，即:
        w是网络权重，也可以理解成是函数拟合中的基函数参数:
        Loss(w) = ||r_t + \gamma max_a(Q(s',a', w)) - Q(s, a, w)||^2

        HINT: Deepmind提出的DQN，本身是存在2个打破数据相关性的trick的:
        1. 设置replay buffer，每次从中random一个batch的数据进行利用
        2. 存在两个网络。一个是DQN主网络，输入state输出Q(s,a)向量(对该state所有的action的Q(s,a)，所以输出是个向量)
                另一个是结构、输入输出维度和DQN主网络完全相同的、只有参数不同的Target Network
                用于计算TD target(这也是他使用TD方法来对值函数估计的最明显证据)
                为了提高训练稳定性，每10步把DQN主网络的权重拷贝到target network中。平时targe network的权重fixed。
        本实现中只有1，没有2(为了简便)

    2. 算法的learn过程
        1. 采样一个episode/或多个，放入buffer中
        2. 从buffer中random一个batch的数据，送入网络中进行训练
        3. 不断循环这个过程                
'''

log_file_path = "logs/dqn.train.log"
if os.path.exists(log_file_path) == True:
    os.remove(log_file_path)
if os.path.exists(log_file_path.split("/")[0]) == False:
    os.mkdir(log_file_path.split("/")[0])


class DQNAgent:
    def __init__(self):
        self.gamma = 0.95   # reward decay to return

        self.epsilon =  1   # exploration percent
        self.epsilon_decay = 0.995  # decrease the exploration each step
        self.epsilon_min = 0.01

        self.buffer = deque(maxlen=2000)    # transition replay buffer

        self.cur_epoch = 0
        return
    
    def create_env(self, env_name = "CartPole-v1"):
        try:
            gym.envs.register(
            id='CartPoleExtraLong-v0',
            entry_point='gym.envs.classic_control:CartPoleEnv',
            max_episode_steps=1500,
            reward_threshold=-110.0,
            )
            self.env = gym.make(env_name)
            self.env._max_episode_steps = 50

            self.state_dims = self.env.observation_space.shape[0]
            self.action_dims = self.env.action_space.n
            self.env.reset()
            
        except Exception as e:
            traceback.print_exc(e)
            print("[error] the name of env is %s" % env_name)

    def build_network(self, lr = 0.001):
        '''
            build the network skeleton
        '''
        assert self.action_dims > 0  and self.env is not None and self.state_dims > 0
        # DQN的网络结构和PG中是一样的，输入: state_{state_dims * 1}, 输出: Q(s,a)_{action_dims * 1}
        '''
            input: state_{None, state_dims}
            mid: Dense(32, relu)
            output: action_value_function_{None, action_dims}
        '''
        units = 24
        model = tf.keras.Sequential()
        model.add(Dense(units, input_dim = self.state_dims, activation="relu"))
        model.add(Dense(self.action_dims, activation = "linear"))
        model.compile(loss = "mse", optimizer = Adam(lr = lr))
        self.model = model


    def remember(self, action, state, next_state, reward):
        '''
            store the trainsitions to the buffer
        '''
        # print(type(action))
        # print(action)
        assert type(action) == int and 0 <= action < self.action_dims
        # print(type(state))
        # print(state.shape)
        # print(type(reward))
        assert type(state) == np.ndarray and state.shape == (1, self.state_dims)
        assert type(next_state) == np.ndarray and next_state.shape == (1, self.state_dims)
        assert type(reward) == float

        self.buffer.append((action, state, next_state, reward))
        return 

    def replay(self, batch_size):
        '''
            this function will train our model.
            the name REPLAY, as a special approach when training DQN
            we will maintain a BIG buffer, so many transitions (s,a,r,s') stored
            a batch of trainsitions would be grasped randomly when we train the moel
        '''
        # random samples
        minibatch = random.sample(self.buffer, batch_size)
        for action, state, next_state, reward in minibatch:
            target = reward
            
            if reward > 0:
                # when it is positive, we need to constitute the TD target
                
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            else:
                # when reward is minus, it means "done", and there is no TD target
                pass

            # Loss = ||r + \gamma * max(Q(s',a') + Q(s,a))||
            target_f = self.model.predict(state)    # Q(s, a)的target Q是直接预测出来的
            target_f[0][action] = target            # 唯独target_f[0][action] = target，这个地方要改掉，就是说把这个action的对应的value给修正掉
            self.model.fit(state, target_f, epochs=1, verbose=0)# 每抽一组，就给他修正一次模型，要修正32次
        
        # epsilon decay, decrease the exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return 0
    def get_action(self, state):
        '''
            get action when train:
            when we try to train the model, we will apply the epsilon-greedt policy
            if random() < epsilon, random actions will be chosen for exploration
            otherwise, utilize the policy network for exploitation
        '''
        if state.shape == (self.state_dims, ):
            state = np.reshape(state, [1, self.state_dims])
        assert state.shape == (1, self.state_dims)

        action = -1
        if np.random.rand() < self.epsilon:
            # random 
            action = np.random.randint(self.action_dims)
            # print("random action selected")
        else:
            # network
            q_s_a = self.model.predict(state)
            q_s_a = np.reshape(q_s_a, (self.action_dims, ))


            assert q_s_a.shape == (self.action_dims, )
            action = int(np.argmax(q_s_a, axis = 0))
            # print("network action selected")
            # print(type(action))

        assert type(action) == int
        return action

    def train_one_step(self, batch_size = 32):
        '''
            this function will collect a episode samples, put them to the replay
            buffer the train in each transition step
        '''
        self.cur_epoch += 1

        # init the environment
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_dims])
        train_loss = []
        batch_reward = []

        # loop to the end of episode
        while True:
            # get action
            action = self.get_action(state)
            assert type(action) == int
            # print(len(train_loss))

            # act in the env
            next_state, reward, done, _ = self.env.step(action)
            next_state = np.reshape(next_state, [1, self.state_dims])
            batch_reward.append(reward)


            if done == True:
                reward = 10.0
            
            # put into buffer
            self.remember(action, state, next_state, reward)

            # update state:
            state = next_state

            # train
            if len(self.buffer) > batch_size:
                # buffer长度可以训练
                loss = self.replay(batch_size)
                train_loss.append(loss)
                
            # judge done
            if done == True:
                break

        info = "epoch %d: loss: %.3f avg_return: %.3f eps = %.3f" % \
            (self.cur_epoch, np.mean(train_loss), np.mean(batch_reward), self.epsilon) 
        print(info)
        global log_file_path
        with open(log_file_path, 'a') as f:
            f.write(info + "\n")

        return np.mean(train_loss), np.mean(batch_reward)

    def test(self):
        '''
            this function will try an episode, to show whether the policy is 
            good or not
        '''
        state = self.env.reset()
        while True:
            self.env.render()
            state = np.reshape(state, (1, self.state_dims))
            assert state.shape == (1, self.state_dims)
            q_s_a = self.model.predict(state)
            q_s_a = np.reshape(q_s_a, (self.action_dims, ))
            assert q_s_a.shape == (self.action_dims, )
            action = np.argmax(q_s_a, axis = 0)
            assert type(action) == int or type(action) == np.int64

            state, reward, done, _ = self.env.step(action)
            if done == True:
                self.env.close()
                break

    def load(self, path):
        assert type(path) == str
        try:
            model_return = int(path.split("/")[-1])
        except Exception as e:
            print("parse path fail. the path_dir must be like './models/500'")
        
        self.epsilon = self.epsilon_min
        print(os.getcwd())
        self.model.load_weights(path)
        return model_return

    def save(self, name):
        self.model.save_weights(name)

if __name__ =="__main__":
    tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)
    DQN = DQNAgent()
    DQN.create_env("MountainCar-v0")
    DQN.build_network(lr = 0.001)
    epochs = 100000

    # read saved model
    # read_model_path = "./dqn/models/915/915" # 读取模型用这个
    read_model_path = None    # 从头训练用这个
    if read_model_path is None:
        save_model_threshold = -200 # 保存model的最低return门槛
        max_ret = -200
    else:
        save_model_threshold = DQN.load(read_model_path)
        max_ret = save_model_threshold
        print("load %s succ" % read_model_path)
    ret_threshold = -200
    

    # train out model
    for i in range(epochs):    
        # train 
        loss, ret = DQN.train_one_step()

        # if the return is quite high, save it
        if ret > max_ret and ret > ret_threshold:
            path = "./models_moutaincar/" + str(ret) +"/" + str(ret)
            DQN.save(path)
            print(path + " saved")
            max_ret = ret

            # then test
        DQN.test()