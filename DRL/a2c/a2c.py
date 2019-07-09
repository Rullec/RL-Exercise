import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import gym
import os
import traceback

'''
    下面的代码实现了A2C(Advantage Actor Critic)网络，详细的推导请看"actor-critic(演员-评论家)"

    1. 算法的结构模型:

    2. 算法的learn模型:
    
'''
log_dir = "logs/"
log_name = "a2c.train.log"
log_path = log_dir + log_name

if False == os.path.exists(log_dir):
    os.makedirs(log_dir)
if os.path.exists(log_path):
    os.remove(log_path)

class Actor:
    def __init__(self, units, state_dims, action_dims, lr, epsilon, epsilon_decay, epsilon_min):
        # init var
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.lr = lr

        # build network
        self._build_network(units = units)

        # epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def get_epsilon(self):
        return self.epsilon

    def _build_network(self, units):
        assert self.action_dims > 0 and self.state_dims >0 
        self.input_layer = tf.placeholder(shape = (None, self.state_dims), dtype = tf.float32, name="input_layer")

        self.mid_layer = tf.layers.dense(self.input_layer, units = units, activation = tf.nn.relu)

        # 这一层输出的是各个action的概率
        self.output_layer = tf.layers.dense(self.mid_layer, units = self.action_dims, activation = None)

        # 这一层依照上面的概率进行采样
        self.action_layer = tf.squeeze(tf.random.categorical(logits = self.output_layer, num_samples = 1), axis = 1)    #shape = (?, 1)
        # self.action_layer = tf.random.categorical(logits = self.output_layer, num_samples = 1)    # shape = (?, )
        
        # 计算batchsize个轨迹(traject)的概率
        self.action_ph = tf.placeholder(dtype = tf.int32, shape = (None, ), name = "action_placeholder")
        self.advantage_ph = tf.placeholder(dtype = tf.float32, shape = (None, ), name = "advantage_placeholder")
        action_mask = tf.one_hot(self.action_ph, self.action_dims, name = "action_mask")
        log_trajectory_probs = tf.reduce_sum(action_mask * tf.nn.log_softmax(self.output_layer), axis=1)# 这些轨迹发生的概率的log是什么
        # tf.shape(log_trajectory_probs) = (batch_size, )

        # 计算loss: 需要乘以return
        self.loss = -tf.reduce_mean(self.advantage_ph * log_trajectory_probs)
        
        # train
        self.train = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(self.loss)

        # init
        init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def get_action(self, cur_state, train = True):
        if cur_state.shape == (self.state_dims, ):
            cur_state = np.reshape(cur_state, (1, self.state_dims))
        assert cur_state.shape == (1, self.state_dims)

        # 目前没有epsilon - greedy
        if np.random.rand() < self.epsilon and train == True:
            action = np.random.randint(self.action_dims)
        else:
            action = self.sess.run(self.action_layer, \
            feed_dict={self.input_layer : cur_state})[0]

        if type(action) == np.int64:
            action = int(action)

        assert type(action) == int and 0 <= action < self.action_dims
        return action

    def get_action_prob(self, state):
        assert type(state) == list
        state = np.reshape(np.array(state), (-1, self.state_dims))
        state_num = state.shape[0]

        # get action probability
        action_prob = self.sess.run(self.output_layer, feed_dict={
            self.input_layer : state
            })

        assert action_prob.shape == (state_num, self.action_dims)
        return action_prob

    def train_one_step(self, batch_states, batch_actions, batch_advantage):
        # 以前，这里是进来batch_return进行训练;

        # 但是现在，因为actor-critic本质上就是对这个return进行了修改，改成了优势函数，所以这里的batch_return应该变成advantage了
        for _ in range(len(batch_states)):
            _, loss = self.sess.run([self.train, self.loss], feed_dict={
                                    self.action_ph: batch_actions,
                                    self.advantage_ph: batch_advantage,
                                    self.input_layer : batch_states})

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def save(self, name):
        saver = tf.train.Saver()
        saver.save(self.sess, name)
        # print("model %s saved!" % name)

    def load(self, name):
        saver = tf.train.Saver()
        saver.restore(self.sess, name)
        # print("model %s load!" % name)

class Critic:
    '''
        actor-critic的架构是来源于
        critic的监督是V(s) = Q(s,a)下面的return......但是这并不是一个均值性质的东西啊，怎么能做监督呢?
        我感觉这个地方有问题啊;

    '''
    def __init__(self, units, state_dims, action_dims, lr):
        # init var
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.lr = lr

        # build network
        self._build_network(units = units)

    def _build_network(self, units):
        # critic network
        model = tf.keras.Sequential()
        model.add(Dense(units, input_dim = self.state_dims, activation="relu"))
        model.add(Dense(units, activation="relu"))
        model.add(Dense(self.action_dims, activation = "linear"))    # 输出action value function
        model.compile(loss = "mse", optimizer = Adam(lr = self.lr))
        self.model = model

    def train_one_step(self, episode_state, episode_action, episode_return):
        # assert len(episode_action) == len(episode_next_state)
        assert len(episode_action) == len(episode_state)
        assert len(episode_action) == len(episode_return)
        # assert type(episode_action[0]) == int
        # assert episode_next_state[0].shape == (1, self.state_dims)
        assert type(episode_return[0]) == float
        assert episode_state[0].shape == (self.state_dims, )

        episode_length = len(episode_action)

        # forward propogation and set goal
        episode_state_array = np.reshape(episode_state, (-1, self.state_dims))
        target_q_s_a = self.model.predict(episode_state_array)
        assert target_q_s_a.shape == (episode_length, self.action_dims)
        for i in range(episode_length):
            target_q_s_a_train = target_q_s_a
            target_q_s_a_train[i][episode_action[i]] = episode_return[i]
        
            # 在这里跑了多个epochs，仍然值得怀疑:会不会有问题啊
            # 先改成1，之后做个实验试试
            self.model.fit(x = episode_state_array, y = target_q_s_a, batch_size = episode_length, verbose=0, shuffle=True, epochs=1)
            
    def get_value(self, episode_state):
        assert type(episode_state) == list

        episode_length = len(episode_state)
        
        # predicit model = get Q(s,a)
        episode_state_array = np.array(episode_state).\
            reshape([episode_length, -1])
        values = self.model.predict(episode_state_array)

        assert values.shape == (episode_length, self.action_dims)
        return values

    def save(self, name):
        self.model.save_weights(name)
        # print("model %s saved!" % name)
        # pass
    def load(self, name):
        self.model.load_weights(name)
        # print("model %s load!" % name)

class A2CAgent:
    def __init__(self, learning_rate = 0.001, units = 24):
        self.actor_lr = learning_rate
        self.critic_lr = learning_rate * 2

        # 探索衰减
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.epsilon_greedy = False # 暂时不启用epsilon greedy策略

        # reward -> return discounted
        self.gamma = 0.95

        # create env
        self._create_env()
        
        # build agent and critic network
        self.actor = Actor(units = units, state_dims = self.state_dims\
            , action_dims = self.action_dims, lr = self.actor_lr, epsilon = self.epsilon,
            epsilon_decay = self.epsilon_decay, epsilon_min = self.epsilon_min)

        self.critic = Critic(units = units, state_dims = self.state_dims,\
            action_dims = self.action_dims, lr = self.critic_lr)

    def get_epsilon(self):
        return self.actor.get_epsilon()

    def _create_env(self, env_name = "CartPole-v0"):
        try:
            gym.envs.register(
            id='CartPoleExtraLong-v0',
            entry_point='gym.envs.classic_control:CartPoleEnv',
            max_episode_steps=1500,
            reward_threshold=-110.0,
            )
            self.env = gym.make(env_name)
            self.env._max_episode_steps = 5000

            self.state_dims = self.env.observation_space.shape[0]
            self.action_dims = self.env.action_space.n
            self.env.reset()

        except Exception as e:
            traceback.print_exc(e)
            print("[error] the name of env is %s" % env_name)


    def get_action(self, state):
        assert state.shape == (self.state_dims, )
        action = self.actor.get_action(state)
        assert type(action) == int
        return action

    def learn(self):
        # init env
        state = self.env.reset()

        episode_state = []
        episode_action = []
        episode_reward = []
        episode_next_state = []

        while True:
            # act in the env
            action = self.get_action(state)
            assert type(action) == int
            next_state, reward, done, _ = self.env.step(action)

            # store
            episode_state.append(state)
            episode_action.append(action)
            episode_reward.append(reward)
            episode_next_state.append(next_state)

            # update
            if done == True:
                break
            else:
                state = next_state

        # train model
        self.train_one_step(episode_state, episode_action, episode_reward, episode_next_state)
        
        return len(episode_action), self.get_epsilon()

    def train_one_step(self, episode_state, episode_action, episode_reward, episode_next_state):
        assert type(episode_state) == list
        assert type(episode_action) == list
        assert type(episode_next_state) == list
        assert type(episode_reward) == list
        episode_length = len(episode_action)

        # compute return
        episode_return = self.compute_return(episode_reward)
        assert len(episode_return) == episode_length

        # compute advantage 
        # v(sk) = SUM{ q(sk,ak) * pi(ak | sk) }
        # A(sk,ak) = q(sk,ak) - v(sk)
        # A = [A(s1,a1), A(s2,a2), ... ,A(sn,an)].shape == (episode_length, )
        # 这里的值函数V如何计算?
        value_list = self.critic.get_value(episode_state)
        action_prob = self.actor.get_action_prob(episode_state)

        assert len(value_list) == episode_length
        episode_advantage = (np.array(episode_return) - np.sum(value_list * action_prob, axis = 1)).tolist()
        assert len(episode_advantage) == episode_length

        # use advantage to train actor
        self.actor.train_one_step(batch_actions = episode_action, batch_advantage = episode_advantage,
            batch_states = episode_state)

        # train critic the same as dqn
        self.critic.train_one_step(episode_state = episode_state, episode_action = episode_action,
            episode_return = episode_return)

    def compute_return(self, rewards):
        '''
            compute return from reward
        '''
        assert type(rewards) == list
        ret_list = []
        cur_return = 0
        
        # ret_i = reward_i + gamma * ret_{i+1}
        for i in reversed(rewards):
            cur_return = self.gamma * cur_return + i
            ret_list.append(cur_return)
        ret_list.reverse()

        return ret_list
    
    def test(self):
        state = self.env.reset()
        max_iter = 200

        for _ in range(max_iter):
            self.env.render()
            action = self.actor.get_action(state, train=False)
            state, reward, done,_ = self.env.step(action)
            if done == True:
                break
        self.env.close()

    def load(self, path):
        assert type(path) == str
        try:
            model_return = int(path.split("/")[-1])
        except Exception as e:
            print("parse path fail. the path_dir must be like './models/500'")
 
        self.actor.load(path + ".actor")
        self.critic.load(path + ".critic")
        print("model %s load!" % path)

        return model_return

    def save(self, name):
        self.actor.save(name + ".actor")
        self.critic.save(name + ".critic")
        print("model %s save!" % path)

if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)
    a2c = A2CAgent()
    epochs = 10000


    # read saved model
    
    read_model_path = "./models/81/81" # 读取模型用这个
    # read_model_path = None    # 从头训练用这个
    if read_model_path is None:
        save_model_threshold = 100 # 保存model的最低return门槛
        max_ret = 100
    else:
        save_model_threshold = a2c.load(read_model_path)
        max_ret = save_model_threshold
        print("load %s succ" % read_model_path)

    ret_list = []
    for i in range(epochs):
        ret, eps = a2c.learn()
        ret_list.append(ret)

        if i % 50 == 0:
            mean_list  = np.mean(np.array(ret_list))
            print("epoch %d: return %.2f, eps %.3f " % (i, mean_list, eps))
            ret_list.clear()
            if mean_list > 150:
                a2c.test()
            
            if ret > max_ret:
                path = "./models/" + str(ret) +"/" + str(ret)
                os.makedirs("./models/"+str(ret)+"/")
                a2c.save(path)
                print(path + " saved")
                max_ret = ret

    print("succ")


# import numpy as np

# state_dims = 10
# action_dims = 4
# units = 24
# lr = 0.001

# model = tf.keras.Sequential()
# model.add(Dense(units, input_dim = state_dims, activation="relu"))
# model.add(Dense(action_dims, activation = "linear"))
# model.compile(loss = "mse", optimizer = Adam(lr = lr))

# # generate data
# input_data = []
# for i in range(32):
#     input_data.append(np.random.random_sample([1, state_dims]))
# input_data = np.array(input_data).reshape([32, -1])
# output_data = np.random.random_sample(action_dims)
# print(input_data)
# print(model.predict(input_data))
# action_dims = 3
# log = np.zeros(action_dims)
# pro = [0.1, 0.1, 0.80000]   # 求和必须是1
# for i in range(100000):
#     action = np.random.choice(np.arange(action_dims), 1, p = pro)[0]
#     # print(action)
#     log[action] += 1
# print(log)