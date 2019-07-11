
import tensorflow as tf
import numpy as np
import gym
import time
from collections import deque
import random

# 设置初始化种子
np.random.seed(1)
tf.set_random_seed(1)

# 超参数
max_episodes = 200000
max_ep_steps = 200
lr_a = 1e-3
lr_c = 1e-3
gamma = 0.9
replacement = [
    dict(name='soft', tau = 0.01),
    dict(name='hard', rep_iter_a = 600, rep_iter_c = 500)
][0]# target network的替换策略由2种，hard和run
memory_capacity = 10000
batch_size = 32

render = False
output_graph = True
env_name = "Pendulum-v0"

################################ Actor ##############################
class Actor(object):
    def __init__(self, sess, action_dim, action_bound,\
        learning_rate, replacement):
        '''
        @sess: 传入的tf会话
        @action_dim: action space的维度是多少? 但不是说好的ddpg能处理无限维度问题吗
        @action_bound: action的上下限制
        @learning_rate: 学习率
        @replacement: 替换策略是soft还是Hard
        '''
        self.sess = sess
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement

        with tf.variable_scope("Actor"):
            # input s, output a
            # 创建eval net: 真正用于训练的，policy actor网络
            self.a = self._build_net(S, scope = "eval_net", trainable = True)
        
            # 创建target net: 目标网路，固定不训练，每次只根据replacement策略进行soft udpate或者HARD UPDATE
            self.a_ = self._build_net(S_, scope = "target_net", trainable= False)
        
        # 拿到eval和target网络的参数
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Actor/eval_net")
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Actor/target_net")

        # 生成更新target网络的tensorflow操作
        # 还能这么玩...
        if self.replacement["name"] == "hard":
            # 硬更新 target = soft
            self.t_replace_couter = 0 # 更新计数器
            self.hard_replace = [tf.assign(t, e) for e, t in zip(self.e_params, self.t_params)]
        else:
            # 软更新 target = (1-tau) * target + tau * soft
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau'] * t + self.replacement['tau'] * e))
                                for e, t in zip(self.e_params, self.t_params)]
            

    def _build_net(self, s, scope, trainable):
        '''
        @s: 输入: 其实这是个state的占位符/当然也可能是next_state的占位符
        @scope: 那么scope
        @trainalbe: 这些参数是否是可以训练的
        '''
        with tf.variable_scope(scope):

            # 上面创建了一层dense net
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(s, 30, activation = tf.nn.relu,
                kernel_initializer=init_w, bias_initializer=init_b,
                name="l1", trainable=trainable)

            # 这里又加了一个scope, 然后加了一层tanh: -1~1
            # 然后输出维度为action_dims. 所以一共有2个全连接
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.action_dim,\
                    activation=tf.nn.tanh, kernel_initializer = init_w,
                    bias_initializer = init_b, name = 'a' , trainable = trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name = "scaled_a")
            # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s): # actor是传入state吗
        
        # 进行训练步骤
        self.sess.run(self.train_op, feed_dict={S: s})

        # replacement: hard or soft...
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_couter % self.replacement['rep_iter_a'] ==0:
                self.sess.run(self.hard_replace)
            self.t_replace_couter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]# 增加一个新轴, 1
        action = self.sess.run(self.a, feed_dict= {S: s})[0]# forward propogation
        return action

    def add_grad_to_graph(self, a_grads):
        '''
            在这里: 手动计算da/dtheta, 乘以dq/da，得到最后的训练题都
            初始化优化器，把我们手工计算的梯度传到进去，得到可供sess.run的一个operatin
            "train_op"
        '''
        with tf.variable_scope("policy_grads"):
            # 求梯度?
            self.policy_grads = tf.gradients(ys = self.a, xs = self.e_params, grad_ys = a_grads)
        
        with tf.variable_scope("A_train"):
            opt = tf.train.AdadeltaOptimizer(-self.lr)  # policy gradient是要沿梯度方向上升的
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement

        with tf.variable_scope('Critic'):
            # input (s, a) output q(s,a) in R 建立q网络
            # 为什么他要把这个a的gradient流动给stop掉?
            # 从本质上来说，他后面接的东西不都是不可训练的吗，难道说这里后面接的是actor网络???那也太邪门了..
            self.a = tf.stop_gradient(a)# 外面传进来的这个a，很有可能是主actor网络的输出; 然后梯度在这里
            self.q = self._build_net(S, self.a, 'eval_net', trainable = True)

            # input (s_, a_), output q_ for q_target 建立q target网络
            self.q_ = self._build_net(S_, a_, "target_net", trainable = False)


            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            # 训练critic网络，需要下一时刻的s, 下一时刻的a, 下一时刻的Q；再减去target_Q
            # 也就是构建TD error的过程，然后令TD error最小...
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            # loss = ||targe_q - q||^2
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
        with tf.variable_scope("a_grad"):
            # 这里计算了Q对a的梯度，这个operation将来估计是给actor那边add上的
            # 主要是，这个里面actor和critic之间是有交互的; c
            # 那我之前写的A2C究竟对不对....
            self.a_grads = tf.gradients(self.q, a)[0]
        
        # 更新策略operation
        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e ) 
                                        for t, e in zip(self.t_params, self.e_params)]
    
    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            # critic网络为什么这么复杂?
            # 先有一层l1,因为他接受两个输入，一个state，一个action。所以要有两个权重
            # 而且普遍来说，占位符要放在matmul前面，这样便于处理batch(None, )
            with tf.variable_scope('l1'):
                n_l1 = 30
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer = init_w, trainable = trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_b, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer = init_b, trainable= trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            # 
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer = init_w, bias_initializer = init_b, trainable = trainable)
            
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={
            S: s, self.a : a, R: r, S_: s_
        })
        # replacement
        if self.replacement["name"] == 'soft':
            self.sess.run(self.soft_replacement)
        else :
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1

####################    Memory  #######################
class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.buffer = deque(maxlen = capacity)
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        self.buffer.append((s, a, r, s_))
        self.pointer += 1
    
    def sample(self, n):
        return random.sample(self.buffer, n)
    
    def is_full(self):
        '''
            return True if the buffer is full
        '''
        return self.buffer.maxlen == len(self.buffer)
# 初始化环境, 获取一些环境信息
env = gym.make(env_name)
env = env.unwrapped
env.seed(1)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high

# 三个占位符，分别对应着: state, reward, next_state；
# 每次训练feed_dict就填他们就好了
with tf.name_scope("S"):
    S = tf.placeholder(tf.float32, shape = [None, state_dim], name="s")
with tf.name_scope("R"):
    R = tf.placeholder(tf.float32, [None, 1], name = "r")
with tf.name_scope("S_"):
    S_ = tf.placeholder(tf.float32, shape = [None, state_dim], name = "s_")

sess = tf.Session()

# create actor and critic
actor = Actor(sess, action_dim, action_bound, lr_a, replacement)
critic = Critic(sess, state_dim, action_dim, lr_c, gamma, replacement, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

sess.run(tf.global_variables_initializer())

M = Memory(memory_capacity, dims = 2 * state_dim + action_dim + 1)

if output_graph:
    tf.summary.FileWriter("logs/", sess.graph)

var = 3# control exploration
t1 = time.time()
for i in range(max_episodes):
    s = env.reset()
    ep_reward = 0

    for j in range(max_ep_steps):
        if render:
            env.render()

        # add exploration noise 选中action
        a = actor.choose_action(s)
        # 增加一个噪声，然后切一下(在合理范围内)
        # 这个a_min and a_max应该改良一下
        a = np.clip(np.random.normal(a, var), -2, 2)
        s_, r, done, _ = env.step(a)
        # reward被砍掉了/10
        M.store_transition(s, a, r/10, s_)

        if M.is_full() == True:
            # 只有当满了以后才会开始训练
            # 这个思路其实挺好的，相当于你增大探索嘛`
            var *= 0.9995
            batch_data = M.sample(batch_size)
            state, action,reward,next_state = [],[],[],[]

            for s, a, r, s_ in batch_data:
                state.append(s)
                action.append(a)
                reward.append(r)
                next_state.append(s_)
            state = np.reshape(np.array(state), [batch_size, state_dim])
            next_state = np.reshape(np.array(next_state), [batch_size, state_dim])
            reward = np.reshape(np.array(reward), [batch_size, 1])
            action = np.reshape(np.array(action), [batch_size, action_dim])

            critic.learn(state, action, reward, next_state)
            actor.learn(state)
        
        s = s_
        ep_reward += r
        if j == max_ep_steps - 1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward > -300:
            #     render = True
            break
    
print('Running time: ', time.time()-t1)




            
            
