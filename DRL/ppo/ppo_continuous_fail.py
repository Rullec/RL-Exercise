import tensorflow as tf
import numpy as np
import pickle
import gym
import tensorflow_probability as tfp
import warnings
import scipy
import os
import shutil
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import norm

warnings.filterwarnings("ignore")
'''
    PPO(Proximal Policy Optimization) continuous case
    the only one difference between continuous and discrete case of PPO，
    is the policy network, which it outputs a softmax distribution vector in finite action space
    and a pair of mean and std_var in continuous action space.
        离散情况下，输出一个向量 logits, action ~ Multi(logits)
        连续情况下，输出mean和std, action ~ N(mean, std)
    
    训练policy网络, policy 网络输入state, 输出action的mean和std_var
        定义clipped loss = min(r_t * A_t , clip(r_t, 0.8, 1.2)*A_t)
        其中r_t = \pi(a_t|s_t) / \pi_{old}(a_t|s_t)
        A_t, \pi_{old}由placeholder给出，\pi(a_t |s_t)由神经网络的输出+高斯分布概率密度函数给出
    
    训练value网络, value 网络输入state, 输出V(state)
        令TD error = r + \gamma V(next_state) - state 趋近于0(MSE)
        利用主-target网络技巧
'''
import math

def normcdf(x, mean, sd):
    if type(mean) is not np.ndarray:
        pass
    else:
        pass
def normpdf(x, mean, std):
    if type(mean) is not np.ndarray:
        res = norm.pdf(x, loc = mean, scale = std)
        # print((res, mean, std))
        # assert res < 2.0
        return res
    else:
        assert x.shape == mean.shape == std.shape 
        pdf = np.zeros_like(x)
        for i in range(mean.shape[0]):
            for j in range(mean.shape[1]):
                pdf[i, j] = normpdf(x[i,j], mean[i, j], std[i, j])
        return pdf

def normal_sample(mean, std):
    assert len(mean.shape) == 2
    assert len(std.shape) == 2
    assert mean.shape == std.shape 
    res = np.zeros_like(mean)
    # res_prob = np.zeros_like(mean)
    
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            mean_, std_ = mean[i, j], std[i, j]
            res[i, j] = np.random.normal(mean_, std_)
            # res_prob[i, j] = normpdf(res[i,j], mean_, std_)

    return res

class PPOAgent:
    def __init__(self):
        # init paras
        # 这样的参数有的时候会成功，有的时候会失败
        self.para_operation(mode="init")

        # create env, must be continuous action space
        self.env = self._create_env(self.env_name)
        self.action_dim = self.env.action_space.shape[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.action_high_bound = self.env.action_space.high
        self.action_low_bound = self.env.action_space.low
        assert self.action_high_bound.shape == (self.action_dim, )
        assert self.action_low_bound.shape == (self.action_dim, )

        # build net
        self._build_network()

    def para_operation(self, mode = "init", name = None):
        param_num = 12
        if mode == "init" and name is None:
            self.env_name = "Pendulum-v0"
            self.horizon_T = 32 # timestep segements between training
            self.K = 10         # K epochs in each iter (PPO specified)
            self.eps = 0.2      # the clip epsilon in surrogate objective
            self.lr_a = 0.001  # learning rate for action net
            self.lr_v = 0.002  # learning rate for value net
            self.tau = 0.001    # target net and eval net soft update speed
            self.gamma = 0.9   # reward -> return decay
            self.explore_eps = 0.0 # if rand() < explore_eps, take an random action
            self.explore_eps_decay = 0.99 # decay
            self.explore_eps_min = 0.02 # min threshold
            self.lmbda = 0.95    # lambda in GAE

        elif mode == "load" and name is not None:
            config_path = name + ".conf"
            with open(config_path, "rb") as f:
                para = pickle.load(f)
                assert len(para) == param_num
                self.env_name = para["env_name"]
                self.horizon_T = para["horizon_T"] # timestep segements between training
                self.K = para["K"]         # K epochs in each iter (PPO specified)
                self.eps = para["eps"]    # the clip epsilon in surrogate objective
                self.lr_a = para["lr_a"]  # learning rate for action net
                self.lr_v = para["lr_v"]  # learning rate for value net
                self.tau = para["tau"]    # target net and eval net soft update speed
                self.gamma = para["gamma"]   # reward -> return decay
                self.explore_eps = para["explore_eps"] # if rand() < explore_eps, take an random action
                self.explore_eps_decay = para["explore_eps_decay"] # decay
                self.explore_eps_min = para["explore_eps_min"] # min threshold
                self.lmbda = para["lmbda"]    # lambda in GAE
                print("load conf from %s succ: %s" % (config_path, str(para)))

        elif mode == "save" and name is not None:
            para = {"env_name": self.env_name, "horizon_T" : self.horizon_T, "K" : self.K,
                "eps": self.eps, "lr_a" : self.lr_a, "lr_v" : self.lr_v, "tau" : self.tau,
                "gamma" : self.gamma, "explore_eps" : self.explore_eps, 
                "explore_eps_decay" : self.explore_eps_decay, "explore_eps_min" : self.explore_eps_min,
                "lmbda" : self.lmbda}
            assert len(para) == param_num
            config_path = name + ".conf"
            with open(config_path, "wb") as f:
                pickle.dump(para, f)
                print("save conf to %s succ" % config_path)
        else:
            assert ValueError, "the option is illegal" 

    def get_action_prob(self, mean, std, action):
        with tf.variable_scope("compute_action_prob"):
            dist = tfp.distributions.Normal(loc = mean, scale = std)
            action_prob = dist.prob(action)
            assert self.action_dim == 1
            # 对action，所有超界的都要改掉...
            # 只处理一维情况, action size = 1
            # 如果一个action是5维的，而里面有一个超了，那是不是这个action代表的概率就被修改成1?
            high_prob = 1 - dist.cdf(self.action_high_bound)
            low_prob = dist.cdf(self.action_low_bound)


            # 超过上界的情况处理:
            high_mask = tf.greater(self.action_high_bound, action) # 超越上界的为0
            high_mask_not = tf.logical_not(high_mask) # 超越上界的为1
            high_mask_float = tf.cast(high_mask, tf.float32)
            high_mask_not_float = tf.cast(high_mask_not, tf.float32)
            action_prob = tf.multiply(high_mask_float, action_prob) + tf.multiply(high_mask_not_float, high_prob)

            # 超过下界的情况处理
            low_mask = tf.greater(action, self.action_low_bound)# 超越下界的为0
            low_mask_not = tf.logical_not(low_mask)# 超越下界的为1
            low_mask_float = tf.cast(low_mask, tf.float32)
            low_mask_not_float = tf.cast(low_mask_not, tf.float32)
            action_prob = tf.multiply(low_mask_float, action_prob) + tf.multiply(low_mask_not_float, low_prob)

            return action_prob

    def _build_policy_net(self):
        with tf.variable_scope("policy_net"):
            # 网络结构
            self.state_ph = tf.placeholder(dtype = tf.float32, shape = [ None, self.state_dim], name = "state_ph")
            l1 = tf.layers.dense(inputs = self.state_ph, units = 32, activation = tf.nn.relu, name = "l1")
            # l2 = tf.layers.dense(inputs = l1, units = 64, activation = tf.nn.relu, name = "l2")
            self.mean = tf.multiply(tf.layers.dense(inputs = l1, units = self.action_dim, activation = tf.nn.tanh), self.action_high_bound, name = "mean")
            self.std = tf.layers.dense(inputs = l1, units = self.action_dim, activation = tf.nn.relu, name = "std") + 1e-5
            # self.action_dist = self.get_distribution(self.mean, self.std)
            # action, 以及action在策略下的prob不在此计算

            # 定义loss = min(surr1, surr2)
            # ratio = pi(a|s) / pi_old(a|s)
            self.action_ph = tf.placeholder(dtype = tf.float32, shape = [None, self.action_dim], name = "action_ph")
            self.action_prob_cur = self.get_action_prob(self.mean, self.std, self.action_ph)
            self.action_prob_old_ph = tf.placeholder(dtype = tf.float32, shape = [None, 1], name = "action_prob_old_ph")
            self.ratio = tf.div(self.action_prob_cur + 1e-8, self.action_prob_old_ph + 1e-8, name = "ratio")
            self.advantage_ph = tf.placeholder(dtype = tf.float32, shape = [None, 1], name = "advantage_ph")
            surr1 = self.ratio * self.advantage_ph
            surr2 = tf.clip_by_value(self.ratio, clip_value_max = 1 + self.eps, clip_value_min = 1 - self.eps) * self.advantage_ph
            self.loss = -tf.minimum(surr1, surr2)
            # self.train_agent_op = tf.train.AdamOptimizer(self.lr_a).minimize(self.loss)
            opt = tf.train.AdamOptimizer(self.lr_a)
            grad = opt.compute_gradients(self.loss)
            clipped_grad = [(tf.clip_by_value(grad_, -1, 1), var)  for grad_, var in grad]
            self.train_agent_op = opt.apply_gradients(clipped_grad)
            tf.summary.scalar('loss', self.loss)

    
    def _build_value_net(self):

        # GAE network
        with tf.variable_scope("value_net"):
            # 主value net和 targe value net
            # soft update
            self.input_state_gae_ph = tf.placeholder(dtype = tf.float32, shape = [None, self.state_dim], name = "input_state_gae")
            with tf.variable_scope("eval_net"):
                l1 = tf.layers.dense(inputs = self.input_state_gae_ph, units = 32, activation = tf.nn.relu, name = "l1_gae", trainable = True)
                # l2 = tf.layers.dense(inputs = l1,units = 64, activation = tf.nn.relu, name = "l2_gae", trainable = True)
                self.output_value_gae_eval = tf.layers.dense(inputs = l1, units = 1, activation = None, name = "output_value_gae")

            self.input_state_gae_target_ph = tf.placeholder(dtype = tf.float32, shape = [None, self.state_dim], name = "input_state_gae_target")
            with tf.variable_scope("target_net"):
                l1 = tf.layers.dense(inputs = self.input_state_gae_target_ph, units = 32, activation = tf.nn.relu, name = "l1_gae", trainable = False)
                # l2 = tf.layers.dense(inputs = l1,units = 64, activation = tf.nn.relu, name = "l2_gae", trainable = False)
                self.output_value_gae_target = tf.layers.dense(inputs = l1, units = 1, activation = None, name = 'output_value_gae')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="value_net/eval_net")
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "value_net/target_net")
            replacement = [tf.assign(t, (1 - self.tau) * t + self.tau * e) 
                                for (t, e) in zip(t_params, e_params)]
            self.reward_ph = tf.placeholder(dtype = tf.float32, shape = [None, 1], name = "reward_ph")
            self.loss_value1 = self.reward_ph + self.gamma * self.output_value_gae_target
            self.loss_value2 = self.output_value_gae_eval
            self.loss_value = tf.reduce_mean(tf.math.squared_difference(self.loss_value1, self.loss_value2))
            self.train_value_op = tf.train.AdamOptimizer(self.lr_v).minimize(self.loss_value)
            self.replacement_value_op = replacement

    def _build_network(self):
        assert self.state_dim > 0 and self.action_dim > 0 

        # build policy net
        self._build_policy_net()
        self._build_value_net()

        self.sess = tf.Session()
        self.merged = tf.summary.merge_all()
        self.saver = tf.summary.FileWriter("./logs", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def _create_env(self, env_name):
        env = gym.make(env_name)
        env.reset()
        return env

    def get_action(self, state, test = False):
        # print("begin get")
        assert state.shape[1] == self.state_dim
        length = state.shape[0]
        assert length == 1
        # print("begin 2")
        mean, std = self.sess.run([self.mean, self.std], feed_dict = { self.state_ph : state})
        # print(mean)
        # print(std)
        # sess.run，并且检查所有新增加的continuous policy网络输出shape
        
        # dist = self.get_distribution(mean, std)

        action = None

        # 计算action
        if test is False and np.random.rand() < self.explore_eps:# 加噪声
            action = np.random.random_sample([length, self.action_dim]) * (self.action_high_bound - self.action_low_bound) + self.action_low_bound
        else:# 正经输出
            action = normal_sample(mean, std)
            # print(action)
            assert action.shape == (length, self.action_dim)

        # 计算prob
        # print("mean %s" % str(mean))
        # print("std %s" % str(std))
        # print("action %s" % str(action))
        # print(action)
        action_prob = normpdf(action, mean, std)
        high_cdf = 1 - scipy.stats.norm.cdf(self.action_high_bound, loc = mean[0, 0], scale = std[0, 0])
        low_cdf = scipy.stats.norm.cdf(self.action_low_bound, loc = mean[0, 0], scale = std[0, 0])
        for i in range(action.shape[0]):
            for j in range(action.shape[1]):
                if action[i, j] >= self.action_high_bound:
                    action_prob[i, j] = high_cdf
                elif action[i, j] <= self.action_low_bound:
                    action_prob[i, j] = low_cdf
                # assert action_prob[i, j] < 1.0
        assert action.shape == (length, self.action_dim)
        assert action_prob.shape == (length, 1)
        return action, action_prob

    def train_one_step(self, render = False):
        '''
            run one iteration
        '''
        state = np.reshape(self.env.reset(), (1, self.state_dim))
        done = False
        return_lst = []
        loss_value_lst = []

        while not done:
            state_lst = []
            action_lst = []
            action_prob_lst = []
            next_state_lst = []
            reward_lst = []
            done_lst = []
            
            # collect data
            for _ in range(self.horizon_T):
                if render == True:
                    self.env.render()
                action, action_prob = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, (1, self.state_dim))
                
                state_lst.append(state)
                action_lst.append(action)
                action_prob_lst.append(action_prob)
                reward_lst.append((reward + 8)/ 8)
                next_state_lst.append(next_state)
                done_lst.append(0 if done else 1)

                if done == True:
                    break
                else:
                    state = next_state
            
            # change data shape
            state_lst = np.reshape(np.array(state_lst), (-1, self.state_dim))
            reward_lst = np.reshape(np.array(reward_lst), (-1, 1))
            next_state_lst = np.reshape(np.array(next_state_lst), (-1, self.state_dim))
            action_lst = np.reshape(np.array(action_lst), (-1, self.action_dim))
            action_prob_lst = np.reshape(np.array(action_prob_lst), (-1, 1))
            done_lst = np.reshape(np.array(done_lst), (-1, ))
            length = state_lst.shape[0]
            # print(np.mean(action_lst), end = ' ')

            # compute advantage function with GAE
            advantage_lst = self.compute_advantage(state_lst, reward_lst, next_state_lst, done_lst)

            # 开始训练两个网络
            for i in range(0, self.K, 1):
                # policy
                # print(type(state_lst))
                # print(state_lst.shape)
                # print(type(action_lst))
                # print(action_lst.shape)
                # print(type(action_prob_lst))
                # print(action_prob_lst.shape)
                # print(type(advantage_lst))
                # print(advantage_lst.shape)
                # print(advantage_lst.shape)
                _, mean, std, action_prob_cur, ratio, loss_policy = self.sess.run([self.train_agent_op, self.mean,\
                     self.std, self.action_prob_cur, self.ratio, self.loss], \
                    feed_dict={
                        self.state_ph : state_lst,
                        self.action_ph : action_lst,
                        self.advantage_ph: advantage_lst,
                        self.action_prob_old_ph : action_prob_lst,
                    })
                # print(action_prob_lst)
                # self.sess.run(self.mean, feed_dict={
                #     self.state_ph : state_lst, 
                #     self.advantage_ph: advantage_lst,
                #     self.action_ph : action_lst,
                #     self.action_prob_old_ph : action_prob_lst})
                    
                # self.sess.run(self.train_agent_op, \
                    # feed_dict={self.state_ph : state_lst,self.advantage_ph: advantage_lst,self.action_prob_old_ph : action_prob_lst,self.action_ph : action_lst})
                assert mean.shape  == (length, 1)
                assert std.shape  == (length, 1)
                assert action_prob_cur.shape == (length, 1)
                assert ratio.shape == (length, 1)
                assert loss_policy.shape == (length, 1)
                if self.iter % 20 == 0 and done and i == self.K-1:
                    print("\n")
                    print("action: %s" % str(np.transpose(action_lst)))
                    print("mean: %s" % str(np.transpose(mean)))
                    print("std: %s" % str(np.transpose(std)))
                    print("action_prob_cur: %s" % str(np.transpose(action_prob_cur)))
                    print("action_prob_cur_old: %s" % str(np.transpose(action_prob_lst)))
                    print("ratio: %s" % str(np.transpose(ratio)))
                    print("advantage: %s" % str(np.transpose(advantage_lst)))
                    print("loss_policy: %s" % str(np.transpose(loss_policy)))

                # value
                _, loss_value = self.sess.run([self.train_value_op, self.loss_value], 
                    feed_dict = {
                        self.input_state_gae_ph : state_lst,
                        self.input_state_gae_target_ph : next_state_lst,
                        self.reward_ph : reward_lst
                })
                loss_value_lst.append(loss_value)

            # value update
            self.sess.run(self.replacement_value_op)
            return_lst.append(np.sum(reward_lst * 8 - 8))
            
        self.explore_eps = max(self.explore_eps * self.explore_eps_decay, self.explore_eps_min)

        ret = np.sum(np.reshape(np.array(return_lst), (-1, )))
        print("\riter %d, ret: %.3f, eps: %.3f, value_net_loss: %.3f" % (self.iter, ret, self.explore_eps, np.mean(loss_value_lst)), end = '')
        return ret
        
    def compute_advantage(self, state, reward, next_state, done):
        '''
            compute advantage with GAE
        '''
        episode_len = state.shape[0]
        assert state.shape == (episode_len, self.state_dim)
        assert reward.shape == (episode_len, 1)
        assert next_state.shape == (episode_len, self.state_dim)
        assert done.shape == (episode_len, )

        state_value = self.sess.run(self.output_value_gae_eval, feed_dict={
            self.input_state_gae_ph: state
        })
        next_state_value = self.sess.run(self.output_value_gae_eval, feed_dict={
            self.input_state_gae_ph : next_state
        })

        # compute delta_lst
        delta_lst = []
        for i in range(episode_len):
            delta = reward[i] + self.gamma * next_state_value[i] * done[i] - state_value[i]
            delta_lst.append(delta)

        # compute advantage
        advantage = 0.0
        advantage_lst = []
        for i in reversed(range(episode_len)):
            advantage = self.lmbda * self.gamma * advantage + delta_lst[i]
            advantage_lst.append(advantage)
        advantage_lst.reverse()
        advantage = np.reshape(np.array(advantage_lst), (-1, 1))
        advantage = np.reshape(np.array([float(advantage[i]) for i in range(advantage.shape[0])]), (-1, 1))
        assert advantage.shape == (episode_len, 1)
        # print(advantage)
        return advantage

    def test(self):
        state = self.env.reset()
        while True:
            self.env.render()
            state = [state, state]
            action, _ = self.get_action(np.reshape(state, (2, self.state_dim)), test = True)
            state, reward, done , _ = self.env.step(action)
            if done == True:
                break
            else:
                if abs(abs(action[0, 0]) - 2) < 1e-3:
                    break
        self.env.close()

    def learn(self, iters = 3000):
        '''
            for each iter 1-N
                (you can also try A3C parallel) multi-workers
                collect T transitions

                train K epochs(thanks to the clipped objective)
        '''
        ret_lst = []
        print_interval = 10
        test_threshold = -1000
        save_threshold = -1000
        self.iter = 0
        while self.iter < iters:
            avg_ret = self.train_one_step()
            ret_lst.append(avg_ret)
            if self.iter % print_interval == 0 and self.iter >0:
                print(", avg_ret: %.3f" % (np.mean(ret_lst[:-print_interval])))

            self.iter += 1

    def save(self, name):
        # model save 
        saver = tf.train.Saver()
        saver.save(self.sess, name)
        self.para_operation(mode = "save", name = name)

        # remember to print log
        print("model saved as %s" % name)
        pass

    def load(self, name):
        # model and paras load
        saver = tf.train.Saver()
        saver.restore(self.sess, name)

        self.para_operation(mode = "load", name = name)
        # remember to print log
        pass
    
if __name__ =="__main__":
    if os.path.exists("logs"):
        shutil.rmtree("logs")
        os.mkdir("logs")
    agent = PPOAgent()
    # agent.load("CartPole-v1/22.523809523809526")
    agent.learn()
    # agent.test()