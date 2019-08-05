import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt

def get_distribution(action_dim, loc, scale):
    if action_dim == 1:
        return tf.distributions.Normal(loc, scale)
    else:
        return tf.contrib.distributions.MultivariateNormalDiag(loc, scale)
# reshape
reshape = lambda lst, dim : np.reshape(np.array(lst), (-1, dim))
print_transpose_str = lambda name, var: print("%s: %s" % (name, str(np.transpose(var))))
shuffle = lambda array, p : array[p]

class PPOAgent(object):
    def __init__(self, env_name = "Pendulum-v0"):
        
        # params init
        self.n_hiddens = 64
        self.lr_c = 0.01
        self.lr_a = 0.005
        self.gamma = 0.99   # reward to return
        self.epsilon_ratio = 0.2  # clipped the surrogate function
        self.K = 20
        self.max_step = 200
        self.batch_size = 32

        # create env
        self.env = self._create_env(env_name)

        # build network
        self._build_network()

    def _build_network(self):
        self._build_policy_net()
        self._build_value_net()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        

    def compute_action_prob_in_loss(self):
        # self.mean, self.sigma, self.action_ph
        prob = tf.exp(-tf.square(self.action_ph - self.mean) / 
                                     (2.0 * tf.square(self.sigma)))
        prob = prob / (tf.sqrt(2 * np.pi) * self.sigma)
        return prob

    def _build_policy_net(self):
        init_w = tf.random_normal_initializer(0.0, 0.3)
        init_b = tf.constant_initializer(0.1)

        with tf.variable_scope("policy_net"):
            # build policy net
            self.state_ph = tf.placeholder(dtype = tf.float32, shape = [None , self.state_dims], name = "state_ph")
            self.l1 = tf.layers.dense(units = self.n_hiddens,inputs = self.state_ph, activation = tf.nn.relu, name ="l1",
                kernel_initializer = init_w, bias_initializer = init_b)
            # self.l2 = tf.layers.dense(units = self.n_hiddens,inputs = self.state_ph, activation = tf.nn.relu, name ="l2")
            # self.l3 = tf.layers.dense(units = self.n_hiddens,inputs = self.state_ph, activation = tf.nn.relu, name ="l3")
            self.mean = tf.multiply(tf.layers.dense(units = self.action_dims, inputs = self.l1, activation = tf.nn.tanh, \
                name = "mean", kernel_initializer = init_w, bias_initializer = init_b), self.action_upper_bound)
            self.sigma = tf.add(tf.layers.dense(units = self.action_dims, inputs = self.l1, activation = tf.nn.softplus,\
                name = "sigma", kernel_initializer = init_w, bias_initializer = init_b), 1e-3)
            self.dist = get_distribution(self.action_dims, self.mean, self.sigma)
            self.action = self.dist.sample()    # action
            self.action_prob = self.get_action_prob(self.dist, self.action) # action prob

            # build loss
            self.action_ph = tf.placeholder(dtype = tf.float32, shape = [None, self.action_dims], name = "action_ph")
            self.action_prob_old_ph = tf.placeholder(dtype = tf.float32, shape = [None, 1], name = "action_prob_old_ph")
            self.advantage_ph = tf.placeholder(dtype = tf.float32, shape = [None, 1], name = "advantage_ph")
            # 然后要获取现在的prob
            self.action_prob_cur = self.compute_action_prob_in_loss()
            self.ratio = tf.exp(tf.log(self.action_prob_cur + 1e-6) - tf.log(self.action_prob_old_ph + 1e-6))
            self.surr1 = self.ratio * self.advantage_ph
            self.surr2 = tf.clip_by_value(self.ratio, clip_value_min = 1 - self.epsilon_ratio,\
                 clip_value_max = 1 + self.epsilon_ratio) * self.advantage_ph
            self.loss_policy = tf.reduce_mean(-tf.minimum(self.surr1, self.surr2))
            self.train_policy_step = tf.train.AdamOptimizer(self.lr_a).minimize(self.loss_policy)

    def get_action_prob(self, dist, action):
        action_prob = dist.prob(action)

        # # high cdf and low cdf
        high_prob = 1 - dist.cdf(self.action_upper_bound)
        low_prob = dist.cdf(self.action_lower_bound)

        # 上界情况处理
        high_mask = tf.greater(self.action_upper_bound, action) # 超越上界的为0
        high_mask_not = tf.logical_not(high_mask) # 超越上界的为1
        high_mask_float = tf.cast(high_mask, tf.float32)
        high_mask_not_float = tf.cast(high_mask_not, tf.float32)
        action_prob = tf.multiply(high_mask_float, action_prob) + tf.multiply(high_mask_not_float, high_prob)

        # 超过下界的情况处理
        low_mask = tf.greater(action, self.action_lower_bound)# 超越下界的为0
        low_mask_not = tf.logical_not(low_mask)# 超越下界的为1
        low_mask_float = tf.cast(low_mask, tf.float32)
        low_mask_not_float = tf.cast(low_mask_not, tf.float32)
        action_prob = tf.multiply(low_mask_float, action_prob) + tf.multiply(low_mask_not_float, low_prob)

        return action_prob

    def _build_value_net(self):
        with tf.variable_scope("value_net"):
            self.state_value_ph = tf.placeholder(dtype = tf.float32, shape = [None, self.state_dims], name = "state_value_ph")
            l1 = tf.layers.dense(units = self.n_hiddens, inputs = self.state_value_ph, activation = tf.nn.relu, name = "l1")
            self.value = tf.layers.dense(units = 1, inputs = l1, activation = None, name = "value")

            self.value_target_ph = tf.placeholder(dtype = tf.float32, shape = [None, 1], name = "value_target_ph")
            self.loss_value = tf.reduce_mean(tf.squared_difference(self.value, self.value_target_ph))
            self.train_value_step = tf.train.AdamOptimizer(self.lr_c).minimize(self.loss_value)

    def _create_env(self, env_name):
        env = gym.make(env_name)
        self.state_dims = env.observation_space.shape[0]
        self.action_dims = env.action_space.shape[0]
        self.action_upper_bound = env.action_space.high
        self.action_lower_bound = env.action_space.low
        return env

    def get_action(self, state):
        assert state.shape == (1, self.state_dims)
        action, action_prob = self.sess.run([self.action, self.action_prob], feed_dict={self.state_ph : state})

        return action, action_prob

    def compute_target_value(self, reward_lst, final_state):
        assert final_state.shape == (1, self.state_dims)
        # compute value function by Monte-Carlo Method
        v_final = self.sess.run(self.value, feed_dict = {self.state_value_ph : final_state})
        cur_value = v_final
        target_lst = []
        for reward in reward_lst[::-1]:
            cur_value = reward + self.gamma * cur_value 
            target_lst.append(cur_value)
        target_lst.reverse()
        return target_lst

    def train_one_step(self):
        
        state = self.env.reset()
        done = False
        total_reward = 0
        sum_value_loss = 0

        step = 0
        while step < self.max_step and not done:
            state_lst, action_lst, reward_lst, next_state_lst, action_prob_lst = [], [], [], [], []

            for i in range(self.batch_size):
                state = np.reshape(state, (1, self.state_dims))
                action, action_prob = self.get_action(state)
                step+=1
                next_state, reward, done , _ = self.env.step(action)

                # reshape
                state_lst.append(state), action_lst.append(action), reward_lst.append((8+reward)/8),\
                next_state_lst.append(next_state), action_prob_lst.append(action_prob)

                if done :
                    break
                else:
                    state = next_state
                

            # 估计值函数: 使用MC法
            target_value_lst = self.compute_target_value(reward_lst, reshape(next_state, self.state_dims))
            # print(target_value_lst)
            state_lst, action_lst, target_value_lst, action_prob_lst, reward_lst = reshape(state_lst, self.state_dims),\
                reshape(action_lst, self.action_dims), reshape(target_value_lst, 1), reshape(action_prob_lst, 1), reshape(reward_lst, 1)

            # shuffle 
            p = np.random.permutation(state_lst.shape[0])
            state_lst, action_lst, target_value_lst, action_prob_lst, reward_lst = shuffle(state_lst, p),\
                shuffle(action_lst, p), shuffle(target_value_lst, p), shuffle(action_prob_lst, p), shuffle(reward_lst, p)
            
            # train
            for j in range(self.K):
                ## train value net (critic)
                # print(state_lst.shape)
                _, value_net_loss = self.sess.run([self.train_value_step, self.loss_value], feed_dict={
                    self.state_value_ph : state_lst,
                    self.value_target_ph: target_value_lst
                })

                ## train policy (actor)
                _, ratio, mean, sigma, action, surr1 = self.sess.run([self.train_policy_step, self.ratio, self.mean, self.sigma, self.action, self.surr1], feed_dict={
                    self.state_ph : state_lst,
                    self.action_ph : action_lst,
                    self.action_prob_old_ph : action_prob_lst,
                    self.advantage_ph: target_value_lst
                })

                # 后处理
                sum_value_loss += value_net_loss
                if j == self.K-1 and done :
                    print_transpose_str("ratio", ratio)
                    print_transpose_str("mean", mean)
                    print_transpose_str("sigma", sigma)
                    print_transpose_str("action", action)
                    print_transpose_str("surr1", surr1)
                    assert surr1.shape == (state_lst.shape[0], 1)
            
            # 后处理
            total_reward += np.sum(reward_lst * 8 - 8)
        print("iter %d, ret: %.3f, value_net_loss: %.3f " % (self.iter, total_reward , sum_value_loss / step))
        return total_reward
    
    def learn(self, iters = 15000):
        self.iter = 0
        ret_lst = []
        plt.ion()
        while self.iter < iters:
            ret_lst.append(self.train_one_step())
            self.iter += 1

            # display
            plt.plot(ret_lst)
            plt.pause(1e-5)
            plt.cla()

if __name__ == "__main__":
    agent = PPOAgent()
    agent.learn()