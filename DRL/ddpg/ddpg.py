import tensorflow as tf
import numpy as np
from collections import deque
import gym
import random

'''
DDPG算法 Deep Deterministic Policy Gradient
深度-确定性策略梯度算法;从名字中我们可以看出，他属于PG方法的一种()

'''
'''
    调了好久，不work，开始逐行抄代码，检测问题
    1. example中用的是elu而非relu。elu可以产生负的输出: 改了以后没有变化
    2. 在训练actor网络时，他把输入的梯度除以了batch size，即dqda/dqda.shape[0]: 改了以后没有变化
    3. 对于value network，输入是state和action，我是直接就合成一个输入然后接网络，他是分别先接了一层权重之后再训练的：改了以后没有变化
    4. placeholder中的shape应该使用[]而非():改了以后试试看： 改了以后不管用...
    5. 如果以tanh(-1 - 1)为激活函数输出的话，那么action的输出是非法的, 改成sigmoid（0-1）之后: 不管用
    6. 我的replay策略，将一半改为一个batch充满了就好，试试看：不管用
    7. 把init_w和init_b改成contrib中的，试试看: 不管用
    8. 突然发现: get target action中，我拿的是eval，这是一个大错误，改了以后试试: 不管用
    9. 噪声和探索: 有人说最好使用paramter / action space噪声，而非之前那样随episode提升的探索。管用了！
    终于work了!
'''
class Actor:
    def __init__(self, sess, state_dim, action_dim, action_high_bound, action_low_bound):
        # hyper parameter

        self.lr_a = 0.0001
        self.tau = 0.001

        # noise paras
        self.noise = 0
        self.theta = 0.15
        self.sigma = 0.2

        self.cur_epoch = 0
        self.sess = sess
        self.state_dims = state_dim
        self.action_dims = action_dim
        self.action_high_bound = action_high_bound
        self.action_low_bound = action_low_bound

        self._build_network()
        
    def _build_network(self):
        '''
            输入: state
            输出: action
        '''
        with tf.variable_scope("Actor"):
            self.input_state = tf.placeholder(dtype = tf.float32, shape = [None, self.state_dims], name = "input_state")

            # 输入两个，完成
            self.output_eval_action = self._build_single_net(inputs = self.input_state, scope_name = "eval_net", trainable = True)
            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "Actor/eval_net")
            self.output_target_action = self._build_single_net(inputs = self.input_state, scope_name = "target_net", trainable = False)
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "Actor/target_net")

            # 定义训练过程
            '''
            定义loss(策略梯度, 训练步骤train_op)
            '''
            self.dqda = tf.placeholder(dtype = tf.float32, shape = [None, self.action_dims], name = "dqda_ph")
            
            with tf.variable_scope("policy_gradient"):
                self.policy_grads = tf.gradients(ys = self.output_eval_action, xs = self.e_params, grad_ys = -self.dqda)
            with tf.variable_scope("train"):
                opt = tf.train.AdamOptimizer(self.lr_a)
                self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))

            # 参数替换
            self.replacement = [tf.assign(t, (1-self.tau) * t + self.tau * e)
                                for (t, e) in zip(self.t_params, self.e_params)]
            
    def _build_single_net(self, inputs, scope_name, trainable):
        '''
            本函数将会定义2个FC层，用作单个的actor网络
        '''
        init_w = tf.random_normal_initializer(0.0, 0.3)
        init_b = tf.constant_initializer(0.1)
        # init_w = tf.contrib.layers.variance_scaling_initializer()
        # init_b = tf.contrib.layers.l2_regularizer(0.01)

        with tf.variable_scope(scope_name):
            l1 = tf.layers.dense(inputs = inputs, units = 32, activation = tf.nn.elu, kernel_initializer = init_w,
            bias_initializer = init_b, trainable = trainable, name = "l1")
            l2 = tf.layers.dense(inputs = l1, units = 64, activation = tf.nn.elu, kernel_initializer = init_w,
            bias_initializer = init_b, trainable = trainable, name = "l2")
            # tanh是 -1 - 1，如果用哪个的话，就会出错，会把输出按-6 - 2来算，那样一定是非法的。
            before_output = tf.layers.dense(inputs = l2, units = 1, activation = tf.nn.tanh, kernel_initializer = init_w,
            bias_initializer = init_b, trainable= trainable, name = "before_output")
            # the output must be scaled to a proper range
            # 假设action是对称的 -1 - 1 -> -2 - 2
            output = tf.multiply(before_output, (self.action_high_bound - self.action_low_bound)/2, name="output")
        return output
    
    def get_target_action(self, state):
        '''
            获得target网络的输出
        '''
        assert state.shape[1] == self.state_dims
        actions = self.sess.run(self.output_target_action, feed_dict = {
            self.input_state: state
        })

        assert actions.shape[1] == self.action_dims
        
        
        for action in actions:
            assert action >= self.action_low_bound-1e-5 and action <= self.action_high_bound + 1e-5
        return actions

    # def get_action(self, state, random_prob = None):
    def get_action(self, state):
        '''
            获得action，做正向的输出
            random_prob: 有多大的可能性进行random? 如果完全前馈输出的话，就设置为None
        '''
        action = self.sess.run(self.output_eval_action, feed_dict={
            self.input_state : state
        })

        # add noise
        noise = np.random.normal(0, 0.3)
        # self.noise += -self.theta * self.noise + self.sigma * np.random.randn()
        action = np.clip(action + noise, self.action_low_bound, self.action_high_bound)

        assert action.shape == (state.shape[0], self.action_dims)
        return action
        

    def train(self, state, dqda):
        '''
            进行actor网络处理
        '''
        assert state.shape[1] == self.state_dims
        assert dqda.shape[1] == self.action_dims
        assert state.shape[0] == dqda.shape[0]

        # 训练
        self.sess.run(self.train_op, feed_dict = {
            self.input_state: state,
            self.dqda: dqda
        })
        return

    def replace_target(self):
        self.sess.run(self.replacement) 

class Critic:
    def __init__(self, sess, state_dim, action_dim):
        self.lr_c = 0.001
        self.tau = 0.001
        self.sess = sess
        self.state_dims = state_dim
        self.action_dims = action_dim
        self.gamma = 0.99

        self.build_network()

    def build_network(self):
        '''
            输入: state and action
            输出: q值s
            loss = r + gamma * Q'(s_, a_) - Q(s, a)
        '''
        
        with tf.variable_scope("Critic"):
            self.input_state = tf.placeholder(dtype = tf.float32, shape = [None, self.state_dims], name = "input_state")
            self.input_action = tf.placeholder(dtype = tf.float32, shape = [None, self.action_dims], name = "input_action")

            # 建立target网络 和 eval网络
            self.output_eval_q = self._build_single_net(self.input_state, self.input_action, scope_name = "eval_net", trainable = True)
            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "Critic/eval_net")
            self.output_target_q = self._build_single_net(self.input_state, self.input_action, scope_name = "target_net", trainable = False)
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "Critic/target_net")

            # 建立loss函数
            self.reward_ph = tf.placeholder(dtype = tf.float32, shape = None, name = "reward_ph")
            self.q_s_a_target_next = tf.placeholder(dtype = tf.float32, shape = [None, self.action_dims], name = "q_s_a_target_next")
            loss = tf.squeeze(tf.squared_difference(self.reward_ph + self.gamma * self.q_s_a_target_next, self.output_eval_q))
            self.train_op = tf.train.AdamOptimizer(self.lr_c).minimize(loss)
            
            # 求dqda
            self.dqda = tf.gradients(ys = self.output_eval_q, xs = self.input_action)

            # 参数替换
            self.replacement = [tf.assign(t, (1-self.tau) * t + self.tau * e)
                                for (t, e) in zip(self.t_params, self.e_params)]
            
    def _build_single_net(self, input_state, input_action, scope_name, trainable):
        init_w = tf.random_normal_initializer(0.0, 0.3)
        init_b = tf.constant_initializer(0.1)
        # init_w = tf.contrib.layers.variance_scaling_initializer()
        # init_b = tf.contrib.layers.l2_regularizer(0.01)

        with tf.variable_scope(scope_name):
            # 定义两层网络
            # input_connect = tf.concat([input_state, input_action], axis = 1, name="input_connected")
            l1_state = tf.layers.dense(inputs = input_state, units = 32, activation = tf.nn.elu,
                kernel_initializer = init_w, bias_initializer= init_b, trainable = trainable,
                name = "l1_state")

            l1_action = tf.layers.dense(inputs = input_action, units = 32, activation = tf.nn.elu,
                kernel_initializer = init_w, bias_initializer = init_b, trainable = trainable,
                name = "l1_action")
            l1 = tf.concat([l1_action, l1_state], axis = 1, name = "l1_connected")
            l2 = tf.layers.dense(inputs = l1, units = 128, activation = tf.nn.elu,
                kernel_initializer = init_w, bias_initializer = init_b, trainable = trainable,
                name = "l2")
            
            output = tf.layers.dense(inputs = l2, units = self.action_dims, activation = None,
                kernel_initializer = init_w, bias_initializer = init_b, trainable = trainable,
                name = "output")

        return output

    def train(self, state, action ,reward, q_target_next):
        assert state.shape[1] == self.state_dims
        assert action.shape[1] == self.action_dims
        assert state.shape[0] == action.shape[0]
        assert reward.shape == (state.shape[0], )
        assert q_target_next.shape == (state.shape[0], self.action_dims)

        loss = self.sess.run(self.train_op, feed_dict={
            self.input_action: action,
            self.input_state: state,
            self.reward_ph : reward,
            self.q_s_a_target_next: q_target_next
        })

    def replace_target(self):
        # 参数替换
        self.sess.run(self.replacement)
        # print("train loss: {}".format(loss))

    def get_target_q_s_a(self, state, action):
        assert state.shape[0] == action.shape[0]
        assert state.shape[1] == self.state_dims
        assert action.shape[1]== self.action_dims

        q = self.sess.run(self.output_target_q, feed_dict = {
            self.input_action: action,
            self.input_state : state
        })
        q = np.reshape(q, [state.shape[0], self.action_dims])
        assert q.shape == (state.shape[0], self.action_dims)
        return q

    def get_dqda(self, state, action):
        assert state.shape[0] == action.shape[0]
        assert state.shape[1] == self.state_dims
        assert action.shape[1]== self.action_dims
        # print(state.shape)
        # print(action.shape)
        dqda = self.sess.run(self.dqda, feed_dict = {
                self.input_action: action,
                self.input_state: state,
            })
        # print(len(dqda))
        dqda = np.reshape(np.array(dqda), [state.shape[0], self.action_dims])
        assert dqda.shape == (state.shape[0], self.action_dims)
        return dqda

class DDPGAgent:
    def __init__(self):

        # 环境
        self.env = self.create_env() 

        # buffer
        self.buffer_maxlen = 10000
        self.buffer = deque(maxlen = self.buffer_maxlen)

        sess = tf.Session()
        self.cur_epoch = 0

        # 创建actor and critics
        self.actor = Actor(sess, self.state_dims, self.action_dims, self.action_high_bound, self.action_low_bound)
        self.critic = Critic(sess, self.state_dims, self.action_dims)

        # 初始化s
        self.writer = tf.summary.FileWriter("./logs", sess.graph)
        sess.run(tf.global_variables_initializer())

        # 当前第几次训练?
        self.cur_iter = 0
        self.total_explore_iter = 10000

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
        except Exception:
            assert 0 == 1, "error create env"
        
    def remember(self, info_list):
        assert len(info_list) == 4
        self.buffer.append(info_list)

    def replay(self, batch_size):
        # 先random
        if len(self.buffer) < batch_size:
            return
        # if self.buffer_maxlen > 2 * len(self.buffer):
        #     return
        datas = random.sample(self.buffer, batch_size)
        batch_state, batch_action, batch_reward, batch_next_state = [], [], [], []
        for state, action, reward, next_state in datas:
            batch_action.append(action)
            batch_state.append(state)
            batch_reward.append(reward)
            batch_next_state.append(next_state)

        batch_state = np.reshape(np.array(batch_state), [batch_size, self.state_dims])
        batch_action = np.reshape(np.array(batch_action), [batch_size, self.action_dims])
        batch_reward = np.reshape(np.array(batch_reward), [batch_size])
        batch_next_state = np.reshape(np.array(batch_next_state), [batch_size, self.state_dims])

        # 数据获得完成，开始训练critic
        batch_next_action = self.actor.get_target_action(batch_next_state)
        assert batch_next_action.shape == (batch_size, self.action_dims)
        # 2. 把next_state和next_action送到target critic里，得到输出q'(s_,a_)
        batch_q_s_a_next = self.critic.get_target_q_s_a(batch_next_state, batch_next_action)
        assert batch_q_s_a_next.shape == (batch_size, self.action_dims)
        # 3. 然后把他们一起送到critic.learn里面去，进行一次训练，顺便计算一次dqda的梯度
        self.critic.train(batch_state, batch_action, batch_reward, batch_q_s_a_next)
        dqda = self.critic.get_dqda(batch_state, batch_action)

        # dqda = self.critic.get
        # print(dqda.shape)
        # print(dqda.transpose())
        assert dqda.shape == (batch_size, self.action_dims)
        
        # 开始训练actor
        self.actor.train(batch_state, dqda)

        # 更新两个target
        self.actor.replace_target()
        self.critic.replace_target()

    def learn(self, batch_size = 64):
        self.cur_epoch += 1
        state = self.env.reset()
        states, next_states, rewards, actions = [], [], [], []
        for _ in range(200):
            self.cur_iter += 1
            # action = self.actor.get_action(np.reshape(state, [1, self.state_dims]), 1 - (self.cur_iter / self.total_explore_iter))
            action = self.actor.get_action(np.reshape(state, [1, self.state_dims]))
            action = np.reshape(action, [1])
            next_state, reward, done, _ = self.env.step(action)
            
            # remember
            rewards.append(reward)
            self.remember((state, action, reward, next_state))

            # 更新
            state = next_state

            # 训练
            self.replay(batch_size)

            print("iter %d, epoch %d" % (self.cur_iter, self.cur_epoch))
            # 退出
            if done == True:
                break
        print(", reward %.3f" % np.sum(rewards))

if __name__ == "__main__":
    agent = DDPGAgent()
    for _ in range(1000):
        agent.learn()