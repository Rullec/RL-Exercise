import numpy as numpy
import tensorflow as tf
import gym
import traceback
from collections import deque
import os
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
        - 主网络critic训练: 和传统的Q网络训练的方法是一样的。
    
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

class Actor:
    def __init__(self, units, state_dims, action_dims, replacement_dict, action_low_bound, action_high_bound, lr, epsilon, \
        epsilon_decay, epsilon_min):
        
        # importane variables
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.action_low_bound = action_low_bound    # the action is a real number, the upper bound is XX
        self.action_high_bound = action_high_bound        # and the lower bound is XX
        
        # replacement policy
        self.update_target_way = replacement_dict["name"]
        if self.update_target_way == "hard":
            self.update_target_iter = replacement_dict["rep_iter_actor"]
            self.update_target_iter_cur = 1
        elif replacement_dict["name"] == "soft":
            self.update_target_tau = replacement_dict["tau"]
        else:
            assert ValueError, "the policy is illegal"

        # build network
        self._build_network(units = units, )

    def _build_network(self, units):
        '''
            actor网络:
            input: s, state_dims
            output: a , real number
            arch: 2 FC layers
        '''
        self.input = tf.placeholder(dtype = tf.float32, shape=(None, self.state_dims), name="input_state")
        # 获得外界传来的...dq/da，用于求解策略梯度
        self.dqda = tf.placeholder(dtype = tf.float32, shape = (None, self.action_dims), name ="input_dqda")

        self.eval_output = self._build_single_net(self.input, "eval_net", units)

        # 这是eval网络的参数
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="eval_net")
        
        # 定义loss
        with tf.variable_scope("policy_gradient"):    
            self.policy_grads = tf.gradients(ys = self.dqda, xs = self.e_params, grad_ys = self.dqda)

        with tf.variable_scope("actor_train"):
            opt = tf.train.AdamOptimizer(-self.lr)
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))

        # 定义target 网络        
        self.target_output = self._build_single_net(self.input, "target_net", units)
        
        # target 网络的参数
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_net")

        # 定义参数replace过程(就是用actor参数覆盖target参数)
        self.para_replacement = []
        if self.update_target_way =="hard":
            left = self.update_target_iter_cur % self.update_target_iter
            if left == 0:
                self.para_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        elif self.update_target_way == "soft":
            self.para_replacement = [
                tf.assign(t, (1 - self.update_target_tau) * t + self.update_target_tau * e)
                for t, e in zip(self.t_params, self.e_params)
            ]
        else:
            assert ValueError, "the update way is illegal"

    def _build_single_net(self, input, scope_name, units):
        # input placeholder
        init_w = tf.random_normal_initializer(0.0, 0.5)
        init_b = tf.constant_initializer(0.1)

        with tf.variable_scope(scope_name):
            l1 = tf.layers.dense(inputs = input, 
                units = units,
                activation = tf.nn.relu,
                kernel_initializer = init_w,
                bias_initializer = init_b,
                trainable = True,
                name = "l1",
                )
            before_output = tf.layers.dense(inputs = l1, 
                units = self.action_dims,
                activation = tf.nn.tanh,
                kernel_initializer = init_w,
                bias_initializer = init_b,
                trainable = True,
                name = "before_output")
            # 获得forward propogation的结果以后，网络结构完成
            output = tf.multiply(before_output, (self.action_high_bound - self.action_low_bound)/2 , name="output_action")
        return output
    
    def train(self):
        '''
            this function will train the actor network
        '''
        # 更新参数
        if self.update_target_way == "hard":
            self.update_target_iter_cur += 1
        
        # self.update_target_iter_cur += 1

        pass 

    def get_target_action(self, state):
        pass

    def get_action(self, state):
        pass
    
class Critic:
    '''

    '''
    def __init__(self, units, state_dims, action_dims, lr, gamma, replacement_dict):
        # init var
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.lr = lr
        self.gamma = gamma

        # replacement policy 
        self.update_target_way = replacement_dict["name"]
        if self.update_target_way == "hard":
            self.update_target_iter = replacement_dict["rep_iter_critic"]
            self.update_target_iter_cur = 1
        elif replacement_dict["name"] == "soft":
            self.update_target_tau = replacement_dict["tau"]
        else:
            assert 0==1,"the replacement policy setting is illegal"

        # build network
        self._build_network(units = units)

    def _build_network(self, units):
        '''
            critic网络: 
                输入: state + action
                输出: q(s,a) 一个实数
                loss: 
        '''
        self.input_s = tf.placeholder(dtype = tf.float32, shape = (None, self.state_dims), name = "input_state")
        self.input_a = tf.placeholder(dtype = tf.float32, shape = (None, self.action_dims), name = "input_action")
        # create eval net(主网络)
        self.eval_output = self._build_single_net(self.input_s, self.input_a, units = units, scope_name = "eval_net",
            trainable = True)
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "eval_net")

        # create target net(target 网络，不可训练s)
        self.targe_output = self._build_single_net(self.input_s, self.input_a, units = units, scope_name = "target_net",
            trainable = False)
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "target_net")

        # 定义loss: \reward + \gamma * Q(s_, \mu(s_)) - Q(s_a)
        self.q_next = tf.placeholder(dtype = tf.float32, shape = (None, ), name = "q_next")
        self.reward_ph = tf.placeholder(dtype = tf.float32, shape = (None, ), name = "reward_ph")
        self.loss = self.reward_ph + self.gamma * self.q_next - self.targe_output
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # 定义target replacement
        # 定义参数replace过程(就是用actor参数覆盖target参数)
        self.para_replacement = []
        if self.update_target_way == "hard":
            left = self.update_target_iter_cur % self.update_target_iter
            if left == 0:
                self.para_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        elif self.update_target_way == "soft":
            self.para_replacement = [
                tf.assign(t, (1 - self.update_target_tau) * t + self.update_target_tau * e)
                for t, e in zip(self.t_params, self.e_params)
            ]
        else:
            assert ValueError, "the update way is illegal"
                
    def _build_single_net(self, input_s, input_a, units, scope_name, trainable):
        init_w = tf.random_normal_initializer(0.0, 0.3)
        init_b = tf.constant_initializer(0.1)
        with tf.variable_scope(scope_name):
            with tf.variable_scope("l1"):
                w1_s = tf.get_variable("w1_s", shape = [self.state_dims, units], 
                initializer = init_w, trainable=trainable)
                w1_a = tf.get_variable("w1_a", shape = [self.action_dims, units], 
                initializer = init_w, trainable=trainable)
                b1 = tf.get_variable("b1", shape = [1, units], initializer= init_b,\
                    trainable = trainable)
                l1 = tf.nn.relu(tf.add(tf.matmul(input_s, w1_s) + tf.matmul(input_a, w1_a)) + b1)
            
            with tf.variable_scope("q"):
                output = tf.layers.dense(name = "output", units = 1, inputs = l1,
                    trainable = trainable, activation=None, kernel_initializer = init_w,
                    bias_initializer = init_b)
        
        return output

class DDPGAgent:
    def __init__(self):
        # init variables
        self.gamma = 0.95   # reward decay to return
        self.epsilon =  1   # exploration percent
        self.epsilon_decay = 0.995  # decrease the exploration each step
        self.epsilon_min = 0.01
        self.buffer = deque(maxlen=10000)    # transition replay buffer
        self.lr_a = 0.001   # actor 学习率
        self.lr_c = 0.001   # critic　学习率
        self.cur_epoch = 0  # 初始话epoch个数

        # create env, it must be continuous control problem
        self.env = self.create_env()

        # create replacement policy
        replacement = [
            dict(name='soft', tau = 0.01),
            dict(name='hard', rep_iter_actor= 600, rep_iter_critic=500)
        ][0]

        # build network
        '''
            def __init__(self, units, state_dims, action_dims, replacement_dict,\
                action_low_bound, action_high_bound, lr=0.001, epsilon=1, \
                epsilon_decay=0.9995, epsilon_min = 0.01):
        '''
        self.actor = Actor(32, self.state_dims, self.action_dims, \
            replacement, self.action_low_bound, self.action_high_bound,\
            self.lr_a, self.epsilon, self.epsilon_decay, self.epsilon_min
        )
        self.critic = Critic()

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
            self.action_dims = env.action_space.n
            self.action_high_bound = env.action_space.high
            self.action_low_bound = env.action_space.low
            return env

        except Exception as e:
            traceback.print_exc(e)
            print("[error] the name of env is %s" % env_name)


    def remember(self, state, action, reward, next_state):
        # 这次整个网络都用tensorflow写，要求所有传进来的都是numpy
        assert type(action) == float
        assert state.shape == (self.state_dims, )
        assert next_state.shape == (self.state_dims, )
        assert type(reward) == float

        self.buffer.append((state, action, reward, next_state))

        return 

    def replay(self, batch_size):
        '''
            采一个minibatch然后进行训练
        '''
        pass

    def get_action(self):
        pass

    def learn(self):
        pass
    
    def load(self, name):
        pass

    def save(self, name):
        pass
    
if __name__ == "__main__":
    # print("SUCC")
    agent = DDPGAgent()
    epochs = 10000

    for i in range(epochs):
        agent.learn()
        pass