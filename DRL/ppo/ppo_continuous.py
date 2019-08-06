import shutil
import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
import pickle
import os

# set random seed
np.random.seed(0)
tf.random.set_random_seed(0)

# reshape
reshape = lambda lst, dim : np.reshape(np.array(lst), (-1, dim))
print_transpose_str = lambda name, var: print("%s: %s" % (name, str(np.transpose(var))))
shuffle = lambda array, p : array[p]
read_para = lambda para_dict, name_list: [para_dict[name] for name in name_list]

class PPOAgent(object):
    def __init__(self, env_name = "Pendulum-v0"):
        
        # params init
        self.para_operation(mode = "init")        

        # create env
        self.env = self._create_env(env_name)

        # build network
        self._build_network()

    def para_operation(self, mode = "init", name = None):
        
        hyper_para_list = ["env_name", "n_hiddens", "lr_a", "lr_c", "gamma", "epsilon_ratio", "K", "max_step", "batch_size", "save_threshold"]
        param_num = len(hyper_para_list)
        if mode == "init" and name is None:
            self.env_name = "Pendulum-v0"
            self.n_hiddens = 32
            self.lr_a = 0.00005
            self.lr_c = 0.0001
            self.gamma = 0.9   # reward to return
            self.epsilon_ratio = 0.2  # clipped the surrogate function
            self.K = 10
            self.max_step = 200
            self.batch_size = 32
            self.save_threshold = -500

        elif mode == "load" and name is not None:
            config_path = name + ".conf"
            with open(config_path, "rb") as f:
                load_para_dict = pickle.load(f)
                assert len(load_para_dict) == param_num

                # load params to the class
                for i in load_para_dict:
                    self.__dict__[i] = load_para_dict[i]
                print("load conf from %s succ: %s" % (config_path, str(load_para_dict)))

        elif mode == "save" and name is not None:
            # get the para
            save_para_dict = {}
            for i in hyper_para_list:
                save_para_dict[i] = self.__dict__[i]
            assert len(save_para_dict) == param_num

            # save para dict
            config_path = name + ".conf"
            with open(config_path, "wb") as f:
                pickle.dump(save_para_dict, f)
                print("save conf to %s succ" % config_path)
        else:
            assert ValueError, "the option is illegal" 
            
    def _build_network(self):
        # model input ph
        with tf.variable_scope('input'):
            self.state_ph = tf.placeholder("float", [None, self.state_dims])
        with tf.variable_scope('action'):
            self.action_ph = tf.placeholder(shape=[None, self.action_dims], dtype=tf.float32)
        with tf.variable_scope('target_value'):
            self.target_value_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        with tf.variable_scope('advantages'):
            self.advantage_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32)


        self.value = self.build_critic_net('value_net')
        pi, pi_param = self.build_actor_net('actor_net', trainable= True)
        old_pi, old_pi_param = self.build_actor_net('old_actor_net', trainable=False)
        self.syn_old_pi = [oldp.assign(p) for p, oldp in zip(pi_param, old_pi_param)]
        self.sample_op = tf.clip_by_value(tf.squeeze(pi.sample(1), axis=0), clip_value_min = self.action_lower_bound, clip_value_max = self.action_upper_bound)[0]

        with tf.variable_scope('critic_loss'):
            self.adv = self.target_value_ph - self.value
            self.critic_loss = tf.reduce_mean(tf.square(self.adv))

        with tf.variable_scope('actor_loss'):
            # loss也是一样的
            ratio = (pi.prob(self.action_ph)+1e-6) / (old_pi.prob(self.action_ph) + 1e-6)   #(old_pi.prob(self.a)+ 1e-5)
            pg_losses= self.advantage_ph * ratio
            pg_losses2 = self.advantage_ph * tf.clip_by_value(ratio, 1.0 - self.epsilon_ratio, 1.0 + self.epsilon_ratio)
            self.actor_loss = -tf.reduce_mean(tf.minimum(pg_losses, pg_losses2))

        self.train_op_actor = tf.train.AdamOptimizer(self.lr_a).minimize(self.actor_loss)
        self.train_op_critic = tf.train.AdamOptimizer(self.lr_c).minimize(self.critic_loss)

        self.sess = tf.Session()
        self.saver = tf.summary.FileWriter("./logs", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def build_actor_net(self, scope, trainable):
        with tf.variable_scope(scope):
            ## fc = 3
            dl1 = tf.contrib.layers.fully_connected(inputs=self.state_ph, num_outputs = 16,
                                                                activation_fn=tf.nn.relu,
                                                                scope='dl1')
            dl2 = tf.contrib.layers.fully_connected(inputs=dl1, num_outputs = 8,
                    activation_fn=tf.nn.relu,
                                scope='dl2')

            mu = 2 * tf.contrib.layers.fully_connected(inputs=dl2, num_outputs=self.action_dims,
                                                    activation_fn=tf.nn.tanh,
                                                    trainable = trainable,
                                                   scope='mu')
            sigma = tf.contrib.layers.fully_connected(inputs=dl2, num_outputs=self.action_dims,
                                                       activation_fn=tf.nn.softplus,
                                                       trainable=trainable,
                                                       scope='sigma') + 1e-5

            norm_dist = tf.contrib.distributions.Normal(loc=mu, scale=sigma)

            param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

            return norm_dist, param

    def build_critic_net(self, scope):
        with tf.variable_scope(scope):
            ## fc = 3
            dl1 = tf.contrib.layers.fully_connected(inputs=self.state_ph, num_outputs = 16,
                                                                activation_fn=tf.nn.relu,
                                                                scope='dl1')
            dl2 = tf.contrib.layers.fully_connected(inputs=dl1, num_outputs = 8,
                    activation_fn=tf.nn.relu,
                                scope='dl2')

            value = tf.contrib.layers.fully_connected(inputs=dl2, num_outputs=1,
                                                      activation_fn=None,
                                                      scope='value')
            return value
        
    def _create_env(self, env_name):
        env = gym.make(env_name)
        env.seed(0)
        self.state_dims = env.observation_space.shape[0]
        self.action_dims = env.action_space.shape[0]
        self.action_upper_bound = env.action_space.high
        self.action_lower_bound = env.action_space.low
        return env

    def get_action(self, s):
        assert s.shape == (1, self.state_dims)
        a = self.sess.run(self.sample_op, {self.state_ph: s})
        assert np.isnan(a) == False
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
        return self.sess.run(self.value, {self.s: s})[0, 0]

    def train_one_step(self):    
        state = self.env.reset()
        done = False
        total_reward = 0

        step = 0
        while step < self.max_step and not done:
            state_lst, action_lst, reward_lst, next_state_lst = [], [], [], []

            for i in range(self.batch_size):
                # if self.iter % 100 ==0:
                #     self.env.render()
                state = np.reshape(state, (1, self.state_dims))
                action = self.get_action(state)
                step+=1
                next_state, reward, done , _ = self.env.step(action)

                # reshape
                state_lst.append(state), action_lst.append(action), reward_lst.append((8+reward)/8),\
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
            # print(target_value_lst)
            state_lst_np, action_lst_np, target_value_lst_np,  reward_lst_np = reshape(state_lst, self.state_dims),\
                reshape(action_lst, self.action_dims), reshape(target_value_lst, 1), reshape(reward_lst, 1)

            # train
            self.train(state_lst_np, action_lst_np, target_value_lst_np)

            # 后处理
            total_reward += np.sum(reward_lst_np * 8 - 8)

        print("\riter %d, ret: %.3f " % (self.iter, total_reward ), end = '')
        return total_reward

    def train(self, s, a, r):
        self.sess.run(self.syn_old_pi)
        global_episodes = self.iter

        adv = self.sess.run(self.adv, {self.state_ph: s, self.target_value_ph: r})
        feed_dict_actor = {}
        feed_dict_actor[self.state_ph] = s
        feed_dict_actor[self.action_ph] = a
        feed_dict_actor[self.advantage_ph] = adv
        feed_dict_critic = {}
        feed_dict_critic[self.state_ph] = s
        feed_dict_critic[self.target_value_ph] = r

        [self.sess.run(self.train_op_actor, feed_dict=feed_dict_actor) for _ in range(self.K)]
        [self.sess.run(self.train_op_critic, feed_dict=feed_dict_critic) for _ in range(self.K)]

    def learn(self, iters = 15000):
        self.iter = 0
        ret_lst = []
        print_interval = 20
        test_interval = 100
        while self.iter < iters:
            ret_lst.append(self.train_one_step())

            # print
            if self.iter % print_interval == 0 and self.iter is not 0:
                avg_ret = np.mean(np.array(ret_lst[-print_interval:]))
                print(", avg_ret: %.3f" % avg_ret)
                # print(ret_lst[-print_interval:])

                # save model
                if avg_ret > self.save_threshold:
                    self.test()
                    model_dir = os.path.join("saved_model", self.env_name)
                    if os.path.exists(model_dir) is False:
                        os.makedirs(model_dir)
                    model_name = os.path.join(model_dir, str(float('%.2f' %avg_ret)) + ".ckpt" )
                    self.save(model_name)
                    self.save_threshold = avg_ret
                
            # test model
            if self.iter % test_interval == 0:
                self.test()
            
            self.iter += 1

    def test(self):
        state = self.env.reset()
        reward_lst = []
        while True:
            self.env.render()
            action = self.get_action(np.reshape(state, (1, self.state_dims)))
            state, reward, done , _ = self.env.step(action)
            reward_lst.append(reward)
            if done == True:
                print("[test] return = %.3f" % np.sum(reward_lst))
                break
            
        self.env.close()

    def save(self, name):
        # model save 
        saver = tf.train.Saver()
        saver.save(self.sess, name)
        self.para_operation(mode = "save", name = name)

        # remember to print log
        print("model saved as %s" % name)
        
    def load(self, name, load_conf = True):
        # model and paras load
        saver = tf.train.Saver()
        saver.restore(self.sess, name)

        if load_conf == True:
            self.para_operation(mode = "load", name = name)
        else:
            self.para_operation(mode = "init", name = name)
        # remember to print log
        print("model load from %s" % name)
    

if __name__ == "__main__":
    if os.path.exists("logs"):
        shutil.rmtree("logs")
    agent = PPOAgent()
    
    # agent.load("saved_model/Pendulum-v0/-184.9.ckpt", load_conf=True)
    agent.learn()
    # while True:
    #     agent.test()