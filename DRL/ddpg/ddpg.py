import datetime
# 显示时间
print(datetime.datetime.now().isoformat())

import numpy as np
import tensorflow as tf
from functools import partial

class Actor(object):
    def __init__(self, n_observation, n_action, name = "actor_net"):
        self.n_observation = n_observation
        self.n_action = n_action
        self.name = name
        self.sess = None
        self.build_model()
        self.build_train()
    
    def build_model(self):
        activation = tf.nn.elu
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.01)
        default_dense = partial(tf.layers.dense,\
                                activation=activation,\
                                kernel_initializer=kernel_initializer,\
                                kernel_regularizer=kernel_regularizer)
        with tf.variable_scope(self.name) as scope:
            observation = tf.placeholder(tf.float32,shape=[None,self.n_observation])
            hid1 = default_dense(observation,32)
            hid2 = default_dense(hid1,64)
            action = default_dense(hid2,self.n_action,activation=tf.nn.tanh,use_bias=False)
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=self.name)
        self.observation,self.action,self.trainable_vars = observation,action,trainable_vars
    
    def build_train(self,learning_rate = 0.0001):
        with tf.variable_scope(self.name) as scope:
            action_grads = tf.placeholder(tf.float32,[None,self.n_action])
            var_grads = tf.gradients(self.action, self.trainable_vars, -action_grads)
            train_op = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(var_grads, self.trainable_vars))
        
        self.action_grads = action_grads
        self.train_op = train_op

    def predict_action(self, obs_batch):
        return self.action.eval(session = self.sess, feed_dict= {
            self.observation: obs_batch
        })
    
    def train(self, obs_batch, action_grads):
        batch_size = len(action_grads)
        self.train_op.run(session=self.sess, feed_dict={
            self.observation: obs_batch,
            self.action_grads: action_grads/batch_size
        })
        #
    
    def set_session(self, sess):
        self.sess = sess
    
    def get_trainable_dict(self):
        '''
            获取可以训练的值的字典
        '''
        return {var.name[len(self.name):]: var for var in self.trainable_vars}

class Critic(object):
    def __init__(self, n_observation, n_action, name='critic_net'):
        self.n_observation = n_observation
        self.n_action = n_action
        self.name = name
        self.sess = None
        self.build_model()
        self.build_train()
    def build_model(self):
        activation = tf.nn.elu
        kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
        kernel_regularizer = tf.contrib.layers.l2_regularizer(0.01)
        default_dense = partial(tf.layers.dense,\
                                activation=activation,\
                                kernel_initializer=kernel_initializer,\
                                kernel_regularizer=kernel_regularizer)
        with tf.variable_scope(self.name) as scope:
            observation = tf.placeholder(tf.float32,shape=[None,self.n_observation])
            action = tf.placeholder(tf.float32,shape=[None,self.n_action])
            hid1 = default_dense(observation,32)
            hid2 = default_dense(action,32)
            hid3 = tf.concat([hid1,hid2],axis=1)
            hid4 = default_dense(hid3,128)
            Q = default_dense(hid4,1, activation=None)
            trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = self.name)
            # 他的网络设计和我也不一样: 我是直接合并以后输出，他是分两层做了input以后，再合并到一起
            # 这样明显是更加科学的，否则
        self.observation = observation
        self.action = action
        self.Q = Q
        self.trainable_vars = trainable_vars
    
    def build_train(self,learning_rate=0.001):
        with tf.variable_scope(self.name) as scope:
            Q_expected = tf.placeholder(tf.float32,shape=[None,1])
            loss = tf.losses.mean_squared_error(Q_expected,self.Q)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss)
        self.Q_expected, self.train_op = Q_expected, train_op
        self.action_grads = tf.gradients(self.Q, self.action)[0]

    def predict_Q(self, obs_batch, action_batch):
        return self.Q.eval(session = self.sess, \
            feed_dict = {
                self.observation: obs_batch,
                self.action: action_batch
            })
    
    def compute_action_grads(self, obs_batch, action_batch):
        return self.action_grads.eval(session = self.sess, \
            feed_dict = {
                self.observation:obs_batch,
                self.action: action_batch
            })
    def train(self,obs_batch,action_batch,Qexpected_batch):
        self.train_op.run(session=self.sess,\
                          feed_dict={self.observation:obs_batch,self.action:action_batch,self.Q_expected:Qexpected_batch})
        
    def set_session(self, sess):
        self.sess = sess
    
    def get_trainable_dict(self):
        '''
        这行是比较关键的，主要在于这个冒号有问题
        '''
        return {var.name[len(self.name):]: var for var in self.trainable_vars}

class AsyncNets(object):
    def __init__(self, class_name):
        class_ = eval(class_name)
        self.net = class_(3,1,name=class_name)
        self.target_net = class_(3,1,name='{}_target'.format(class_name))
        self.TAU = tf.placeholder(tf.float32,shape=None)
        self.sess = None
        self.__build_async_assign()
    
    def __build_async_assign(self):
        net_dict = self.net.get_trainable_dict()
        target_net_dict = self.target_net.get_trainable_dict()
        keys = net_dict.keys()
        async_update_op = [target_net_dict[key].assign((1 - self.TAU) * target_net_dict[key] + self.TAU * net_dict[key]) 
                           for key in keys]
        self.async_update_op = async_update_op

    def async_update(self, tau = 0.01):
        self.sess.run(self.async_update_op, feed_dict = {
            self.TAU : tau
        })
    
    def set_session(self, sess):
        self.sess = sess

    def get_subnets(self):
        return self.net, self.target_net


from collections import deque
class Memory(object):
    def __init__(self,memory_size=10000):
        self.memory = deque(maxlen=memory_size)
        self.memory_size = memory_size
    
    def __len__(self):
        return len(self.memory)
    
    def append(self, item):
        self.memory.append(item)
    
    def sample_batch(self, batch_size = 256):
        idx = np.random.permutation(len(self.memory))[:batch_size]
        return [self.memory[i] for i in idx]

def UONoise():
    theta = 0.15
    sigma = 0.2
    state = 0
    while True:
        yield state
        state += -theta * state + sigma * np.random.randn()
    
import gym
from gym import wrappers
max_episode = 500
gamma = 0.99
tau = 0.001
memory_size = 10000
batch_size = 256
memory_warmup = batch_size * 3
max_explore_eps = 100
save_path = "DDPG_net_Class.ckpt"

# build network
tf.reset_default_graph()

# 2 actor nets
actorAsync = AsyncNets("Actor")
actor, actor_target = actorAsync.get_subnets()

# 2 critic nets
criticAsync = AsyncNets("Critic")
critic, critic_target = criticAsync.get_subnets()

# init
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    init.run()
    actorAsync.set_session(sess)
    criticAsync.set_session(sess)
    env = gym.make("Pendulum-v0")
    obs = env.reset()
    iteration = 0
    episode = 0
    episode_score = 0
    episode_steps = 0
    noise = UONoise()
    memory = Memory(memory_size)
    while episode < max_episode:
        print("\riter {}, ep {}".format(iteration, episode), end ='')
        
        # get action
        action = actor.predict_action(np.reshape(obs, [1, -1]))[0]

        # add noise
        if episode < max_explore_eps:
            if np.random.rand() > (episode * 1.0 / max_explore_eps):
                action = np.array([np.random.rand() * 2 - 1])
        
        action *= 2
        next_obs, reward, done, info = env.step(action)
        memory.append([obs, action ,reward, next_obs, done])

        # if iter is bigger than 768, then: begin to shuffle and train
        if iteration >= memory_warmup:
            # 从batch中采样256个
            memory_batch = memory.sample_batch(batch_size)

            # 定义lambda表达式，分别从中提取state, action ,rweard, next_state, done
            extract_mem = lambda k : np.array([item[k] for item in memory_batch])
            obs_batch = extract_mem(0)
            action_batch = extract_mem(1)
            reward_batch = extract_mem(2)
            next_obs_batch = extract_mem(3)
            done_batch = extract_mem(4)

            # 首先用target预测出下一个action，这个和我一样
            action_next = actor_target.predict_action(next_obs_batch)
            # 然后输入action_next和next_state，给出Q_next
            Q_next = critic_target.predict_Q(next_obs_batch,action_next)[:,0]

            # 计算TD target，如果此时batch是结束的，那么...TD target只是当前步骤的收益[这一步和我不一样]
            # Qexpected_batch = reward_batch + gamma*(1-done_batch)*Q_next # target Q value
            # 修改这步会不会有问题呢?[改了以后发现也没什么问题，也可以训练成功]
            Qexpected_batch = reward_batch + gamma*Q_next # target Q value
            Qexpected_batch = np.reshape(Qexpected_batch,[-1,1])
            # train critic 
            # critic完成训练
            critic.train(obs_batch, action_batch, Qexpected_batch)
            # train actor
            # critic计算梯度:这一步和我也是一样的...
            action_grads = critic.compute_action_grads(obs_batch, action_batch)
            actor.train(obs_batch, action_grads)
            # async update
            actorAsync.async_update(tau)
            criticAsync.async_update(tau)
        
        episode_score += reward
        episode_steps += 1
        iteration += 1
        if done:
            print(', score {:8f}, steps {}'.format(episode_score,episode_steps))
#             if episode%5 == 0:
                
#                 Q_check = 
            obs = env.reset()
            episode += 1
            episode_score = 0
            episode_steps = 0
            noise = UONoise()
            if episode%100==0:
                saver.save(sess,save_path)
        else:
            obs = next_obs
env.close()
