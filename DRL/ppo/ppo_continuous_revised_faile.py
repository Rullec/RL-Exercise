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
        self.n_hiddens = 32
        self.lr_a = 0.0001
        self.lr_c = 0.0002
        self.gamma = 0.9   # reward to return
        self.epsilon_ratio = 0.2  # clipped the surrogate function
        self.K = 10
        self.max_step = 200
        self.batch_size = 32

        # create env
        self.env = self._create_env(env_name)

        # build network
        self._build_network()

    def _build_network(self):
        # model input ph
        with tf.variable_scope('input'):
            self.s = tf.placeholder("float", [None, self.state_dims])
        with tf.variable_scope('action'):
            self.a = tf.placeholder(shape=[None, self.action_dims], dtype=tf.float32)
        with tf.variable_scope('target_value'):
            self.y = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        with tf.variable_scope('advantages'):
            self.advantage = tf.placeholder(shape=[None, 1], dtype=tf.float32)


        self.value  = self.build_critic_net('value_net')
        pi, pi_param = self.build_actor_net('actor_net', trainable= True)
        old_pi, old_pi_param = self.build_actor_net('old_actor_net', trainable=False)
        self.syn_old_pi = [oldp.assign(p) for p, oldp in zip(pi_param, old_pi_param)]
        self.sample_op = tf.clip_by_value(tf.squeeze(pi.sample(1), axis=0), clip_value_min = self.action_lower_bound, clip_value_max = self.action_upper_bound)[0]

        with tf.variable_scope('critic_loss'):
            self.adv = self.y - self.value
            self.critic_loss = tf.reduce_mean(tf.square(self.adv))

        with tf.variable_scope('actor_loss'):
            # loss也是一样的
            ratio = pi.prob(self.a) / old_pi.prob(self.a)   #(old_pi.prob(self.a)+ 1e-5)
            pg_losses= self.advantage * ratio
            pg_losses2 = self.advantage * tf.clip_by_value(ratio, 1.0 - self.epsilon_ratio, 1.0 + self.epsilon_ratio)
            self.actor_loss = -tf.reduce_mean(tf.minimum(pg_losses, pg_losses2))

        self.train_op_actor = tf.train.AdamOptimizer(self.lr_a).minimize(self.actor_loss)
        self.train_op_critic = tf.train.AdamOptimizer(self.lr_c).minimize(self.critic_loss)

        self.sess = tf.Session()
        self.saver = tf.summary.FileWriter("./logs", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def build_actor_net(self, scope, trainable):
        with tf.variable_scope(scope):
            dl1 = tf.contrib.layers.fully_connected(inputs=self.s, num_outputs=self.n_hiddens,
                                                    activation_fn=tf.nn.relu,
                                                    trainable = trainable,
                                                    scope='l1')

            mu = 2 * tf.contrib.layers.fully_connected(inputs=dl1, num_outputs=self.action_dims,
                                                    activation_fn=tf.nn.tanh,
                                                    trainable = trainable,
                                                   scope='mu')
            sigma = tf.contrib.layers.fully_connected(inputs=dl1, num_outputs=self.action_dims,
                                                       activation_fn=tf.nn.softplus,
                                                       trainable=trainable,
                                                       scope='sigma')
            norm_dist = tf.contrib.distributions.Normal(loc=mu, scale=sigma)

            param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

            return norm_dist, param

    def build_critic_net(self, scope):
        with tf.variable_scope(scope):
            dl1 = tf.contrib.layers.fully_connected(inputs=self.s, num_outputs=self.n_hiddens,
                                                    activation_fn=tf.nn.relu,
                                                    scope='l1')

            value = tf.contrib.layers.fully_connected(inputs=dl1, num_outputs=1,
                                                      activation_fn=None,
                                                      scope='value')  #[:, 0]  # initializer std 1.0
            #param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            return value
        
    def _create_env(self, env_name):
        env = gym.make(env_name)
        self.state_dims = env.observation_space.shape[0]
        self.action_dims = env.action_space.shape[0]
        self.action_upper_bound = env.action_space.high
        self.action_lower_bound = env.action_space.low
        return env

    def get_action(self, s):
        assert s.shape == (1, self.state_dims)
        a = self.sess.run(self.sample_op, {self.state_ph: s})
        return a

    def compute_target_value(self, reward_lst, final_state):
        assert final_state.shape == (1, self.state_dims)
        # compute value function by Monte-Carlo Method
        v_final = self.sess.run(self.value, feed_dict = {self.state_ph : final_state})
        cur_value = v_final
        target_lst = []
        for reward in reward_lst[::-1]:
            cur_value = reward + self.gamma * cur_value 
            target_lst.append(cur_value)
        target_lst.reverse()
        return target_lst
    
    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.value, {self.state_ph: s})[0, 0]

    def one_epoch(self):
        
        state = self.env.reset()
        done = False
        total_reward = 0
        sum_value_loss = 0

        step = 0
        while step < self.max_step and not done:
            state_lst, action_lst, reward_lst, next_state_lst = [], [], [], []

            for i in range(self.batch_size):
                state = np.reshape(state, (1, self.state_dims))
                action = self.get_action(state)
                step+=1
                next_state, reward, done , _ = self.env.step(action)

                # reshape
                state_lst.append(state), action_lst.append(action), reward_lst.append((8 + reward) /8),\
                next_state_lst.append(next_state)

                if done :
                    summary = tf.Summary()
                    summary.value.add(tag="Rewards/Total_Rewards", simple_value = float(total_reward))
                    summary.value.add(tag="Rewards/Episode_Length", simple_value = float(step))
                    self.saver.add_summary(summary, self.iter)
                    self.saver.flush()
                    break
                else:
                    state = next_state
            
            target_value_lst = self.compute_target_value(reward_lst, reshape(next_state, self.state_dims))

            # reshape
            state_lst_np, action_lst_np, target_value_lst_np,  reward_lst_np = reshape(state_lst, self.state_dims),\
                reshape(action_lst, self.action_dims), reshape(target_value_lst, 1), reshape(reward_lst, 1)
        
            # train
            # train 1: update the old pi policy
            self._train_one_step(state_lst_np, action_lst_np, target_value_lst_np)

            # 后处理
            total_reward += np.sum(reward_lst_np * 8 - 8)
        
        print("\riter %d, ret: %.3f " % (self.iter, total_reward), end = '')

        return total_reward

    def _train_one_step(self, s, a, r):
        
        global_episodes = self.iter

        adv = self.sess.run(self.diff, {self.state_ph: s, self.target_value_ph: r})
        feed_dict_actor = {}
        feed_dict_actor[self.state_ph] = s
        feed_dict_actor[self.action_ph] = a
        feed_dict_actor[self.advantage_ph] = adv
        feed_dict_critic = {}
        feed_dict_critic[self.state_ph] = s
        feed_dict_critic[self.target_value_ph] = r

        [self.sess.run(self.train_op_actor, feed_dict=feed_dict_actor) for _ in range(self.K)]
        [self.sess.run(self.train_op_critic, feed_dict=feed_dict_critic) for _ in range(self.K)]

        # summary = tf.Summary()
        # summary.value.add(tag="Loss/policy", simple_value = float(total_reward))
        # summary.value.add(tag="Loss/value", simple_value = float(step))
                    
        # self.summary_log(self.sess, feed_dict_actor, feed_dict_critic, global_episodes)

        # if self.num_training % 500 == 0:
        #     self.ppo.save_model(sess, saver, global_episodes)

    def learn(self, iters = 15000):
        self.iter = 0
        ret_lst = []
        print_interval = 20

        while self.iter < iters:
            ret_lst.append(self.one_epoch())
            if self.iter % print_interval == 0:
                print(", avg_ret: %.3f" % np.mean(np.array(ret_lst[:-print_interval])))
            self.iter += 1

if __name__ == "__main__":
    agent = PPOAgent()
    agent.learn()