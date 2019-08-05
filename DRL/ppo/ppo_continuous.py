import tensorflow as tf
import numpy as np
import pickle
import gym
import tensorflow_probability as tfp
import warnings
import scipy
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

def normpdf(x, mean, sd):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom

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
            self.horizon_T = 40 # timestep segements between training
            self.K = 5         # K epochs in each iter (PPO specified)
            self.eps = 0.2      # the clip epsilon in surrogate objective
            self.lr_a = 0.0001  # learning rate for action net
            self.lr_v = 0.0002  # learning rate for value net
            self.tau = 0.002    # target net and eval net soft update speed
            self.gamma = 0.995   # reward -> return decay
            self.explore_eps = 3.0 # if rand() < explore_eps, take an random action
            self.explore_eps_decay = 0.99 # decay
            self.explore_eps_min = 0.05 # min threshold
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

    # def _change_action_prob(self, action_dist, action, eps):
    #         low_prob = action_dist.cdf(self.action_low_bound)
    #         up_prob = 1 - action_dist.cdf(self.action_high_bound)
    #         output_action_prob = action_dist.prob(action) + eps
            # 如果action的概率=-2，那么他发生的概率是p(x<-2)
            # # 如果action的概率=2, 那么他发生的概率是p(x>2)
            # # 所有等于其上界的action对应位置的prob全部变成0，然后再用1的相加
            # mask_eq_upper_bound = tf.math.less(tf.abs(self.output_action - self.action_high_bound), tf.ones_like(self.output_action) * eps) # =2的是1
            # mask_eq_lower_bound = tf.math.less(tf.abs(self.output_action - self.action_low_bound), tf.ones_like(self.output_action) * eps) # =-2的是1
            # mask_eq_lower_bound_not = tf.math.logical_not(mask_eq_lower_bound)
            # mask_eq_upper_bound_not = tf.math.logical_not(mask_eq_upper_bound)
            # # change to tf.float32
            # print(type(mask_eq_lower_bound))
            # print(type(mask_eq_lower_bound_not))
            # mask_eq_lower_bound = tf.cast(mask_eq_lower_bound, tf.float32)
            # mask_eq_lower_bound_not = tf.cast(mask_eq_lower_bound_not, tf.float32)
            # mask_eq_upper_bound = tf.cast(mask_eq_upper_bound, tf.float32)
            # mask_eq_upper_bound_not = tf.cast(mask_eq_upper_bound_not, tf.float32)
            # # none
            # output_action_prob = tf.multiply(mask_eq_upper_bound_not, output_action_prob) + tf.multiply(up_prob, mask_eq_upper_bound)
            # output_action_prob = tf.multiply(mask_eq_lower_bound_not, output_action_prob) + tf.multiply(low_prob, mask_eq_lower_bound)
            # return output_action_prob

    def _build_network(self):
        assert self.state_dim > 0 and self.action_dim > 0 
        eps = 1e-6 # for numerial stabily
        # main agent policy network
        with tf.variable_scope("policy_net"):
            # build network
            self.input_state_ph = tf.placeholder(dtype = tf.float32, shape = [None, self.state_dim], name = "input_state")
            l1 = tf.layers.dense(inputs = self.input_state_ph, units = 32, activation = tf.nn.relu, name = "l1")
            l2 = tf.layers.dense(inputs = l1, units = 64, activation = tf.nn.relu, name = "l2")
            # mean在范围内变化, sigma在0-无穷变化
            # self.output_mean = tf.add(tf.multiply(\
            #     tf.layers.dense(inputs = l2, units = self.action_dim, activation = tf.nn.sigmoid),\
            #          self.action_high_bound - self.action_low_bound), self.action_low_bound, name = "l3_mean")
            self.output_mean = tf.layers.dense(inputs = l2, units = self.action_dim, activation = None, name = "l3_mean")
            self.output_std = tf.layers.dense(inputs = l2, units = self.action_dim, activation = tf.nn.softplus, name = "l3_std")
            # 计算prob
            self.action_dist = tfp.distributions.Normal(loc = self.output_mean, scale = self.output_std)
            self.output_action = tf.squeeze(self.action_dist.sample(1), axis = 0)
            self.output_action_prob = self.action_dist.prob(self.output_action) + eps
            # add summary
            tf.summary.histogram('mean', self.output_mean)
            tf.summary.histogram('std', self.output_std)
            # tf.summary.scalar('mean', self.output_mean)
            # tf.summary.scalar('output_action', self.output_action)
            # tf.summary.scalar('output_action_prob', self.output_action_prob)


            # 定义loss
            self.action_ph = tf.placeholder(dtype = tf.float32, shape = [None, 1], name = "action_ph")
            self.action_prob_old_ph = tf.placeholder(dtype = tf.float32, shape = [None, 1], name = "action_prob_old_ph")
            self.advantage_ph = tf.placeholder(dtype = tf.float32, shape = [None, 1], name="advantage_ph")
            self.output_action_prob_cur = self.action_dist.prob(self.action_ph, name = "actrion_prob_cur")
            ratio = tf.div(self.output_action_prob_cur, tf.maximum(self.action_prob_old_ph, eps))

            # 计算surr1和surr2
            surr1 = ratio * self.advantage_ph
            surr2 = tf.clip_by_value(ratio, clip_value_min = 1 - self.eps, clip_value_max = 1 + self.eps) * self.advantage_ph
            loss_src = -tf.minimum(surr1, surr2, name = "clipped_obj")

            hi_mu_loss = tf.square(tf.maximum(0.0, self.output_mean - self.action_high_bound * 1.1))
            lo_mu_loss = tf.square(tf.maximum(0.0, self.action_high_bound * 1.1 - self.output_mean))
            scale_diff = self.action_high_bound - self.action_low_bound
            sigma_bound = tf.square(tf.maximum(0.0, self.output_std - scale_diff * 2.0))
            loss_penalty = tf.reduce_sum(hi_mu_loss + lo_mu_loss + sigma_bound, axis=1)
            loss = loss_src + loss_penalty
            self.train_agent_op = tf.train.AdamOptimizer(self.lr_a).minimize(loss)

            # 所有action超过界限的, prob都被设置为eps，
            # build policy net loss
            # 是否要施加越界惩罚?
            # self.advantage_ph = tf.placeholder(dtype = tf.float32, shape = [None, 1], name = "advantage_ph")
            # self.action_prob_old_ph = tf.placeholder(dtype = tf.float32, shape = [None, 1], name = "action_prob_old_ph")
            

        # GAE network
        with tf.variable_scope("value_net"):
            # 主value net和 targe value net
            # soft update

            self.input_state_gae_ph = tf.placeholder(dtype = tf.float32, shape = [None, self.state_dim], name = "input_state_gae")
            with tf.variable_scope("eval_net"):
                l1 = tf.layers.dense(inputs = self.input_state_gae_ph, units = 32, activation = tf.nn.relu, name = "l1_gae", trainable = True)
                l2 = tf.layers.dense(inputs = l1,units = 64, activation = tf.nn.relu, name = "l2_gae", trainable = True)
                self.output_value_gae_eval = tf.layers.dense(inputs = l2, units = 1, activation = None, name = "output_value_gae")

            self.input_state_gae_target_ph = tf.placeholder(dtype = tf.float32, shape = [None, self.state_dim], name = "input_state_gae_target")
            with tf.variable_scope("target_net"):
                l1 = tf.layers.dense(inputs = self.input_state_gae_target_ph, units = 32, activation = tf.nn.relu, name = "l1_gae", trainable = False)
                l2 = tf.layers.dense(inputs = l1,units = 64, activation = tf.nn.relu, name = "l2_gae", trainable = False)
                self.output_value_gae_target = tf.layers.dense(inputs = l2, units = 1, activation = None, name = 'output_value_gae')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="value_net/eval_net")
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "value_net/target_net")
            replacement = [tf.assign(t, (1 - self.tau) * t + self.tau * e) 
                                for (t, e) in zip(t_params, e_params)]
            self.reward_ph = tf.placeholder(dtype = tf.float32, shape = [None, 1], name = "reward_ph")

            loss = tf.reduce_mean(self.reward_ph + self.gamma * self.output_value_gae_target - self.output_value_gae_eval)
            self.train_value_op = tf.train.AdamOptimizer(self.lr_v).minimize(loss)
            self.replacement_value_op = replacement

        self.sess = tf.Session()
        self.merged = tf.summary.merge_all()
        self.saver = tf.summary.FileWriter("./logs", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def _create_env(self, env_name):
        env = gym.make(env_name)
        env.reset()
        return env

    def get_action(self, state, test = False):
        # sess.run，并且检查所有新增加的continuous policy网络输出shape
        # assert 0 == 1
        
        assert state.shape[1] == self.state_dim
        state_len = state.shape[0]

        # if test is False and np.random.rand() < self.explore_eps:
        #     # 测试
        #     assert state_len == 1
        #     action = np.random.rand() * (self.action_high_bound - self.action_low_bound) + self.action_low_bound
        #     action_mean, action_std = self.sess.run([self.output_mean, self.output_std], feed_dict={
        #         self.input_state_ph : state
        #     })
        #     action_prob = np.reshape(np.array([normpdf(action, action_mean[0], action_std[0])]), (state_len, 1))
        #     action = np.reshape(np.array(action), (state_len, self.action_dim))

        # else:

        '''
            self.input_state_ph = tf.placeholder(dtype = tf.float32, shape = [None, self.state_dim], name = "input_state")
            self.output_mean = tf.add(tf.multiply(\
                tf.layers.dense(inputs = l2, units = self.action_dim, activation = tf.nn.sigmoid),\
                    self.action_high_bound - self.action_low_bound), -self.action_low_bound, name = "l3_mean")
            self.output_std = tf.layers.dense(inputs = l2, units = self.action_dim, activation = tf.nn.softplus, name = "l3_std")
            self.output_action = tf.clip_by_value(action_dist.sample([1]) + eps, clip_value_min = self.action_low_bound, clip_value_max = self.action_high_bound)
            
            self.output_action_prob = tf.math.multiply(valid_array, output_action_prob)
        '''
        mean, std, action, action_prob = self.sess.run([self.output_mean, self.output_std, self.output_action, self.output_action_prob], feed_dict={
            self.input_state_ph : state
        })
        noise = np.random.normal(0, 0.3)
        action += noise
        # print("mean %s, mean shape %s" % (str(mean), str(mean.shape)))
        # print("std %s, std shape %s" % (str(std), str(std.shape)))
        # print("sample_res %s, sample_res shape %s" % (str(sample_res), str(sample_res.shape)))
        # print("action %s, action shape %s" % (str(action), str(action.shape)))
        # print("action_prob %s, action_prob shape %s" % (str(action_prob), str(action_prob.shape)))
        assert mean.shape == (state_len, 1)
        assert std.shape == (state_len, 1)
        assert action.shape == (state_len, self.action_dim)
        assert action_prob.shape == (state_len, 1)
            
        return action, action_prob
        
    def train_one_step(self, render = False):
        '''
            run one iteration
        '''
        state = np.reshape(self.env.reset(), (1, self.state_dim))
        done = False
        return_lst = []

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
                reward_lst.append(reward)
                next_state_lst.append(next_state)
                done_lst.append(0 if done else 1)

                state = next_state
                if done == True:
                    break
            
            # change data shape
            state_lst = np.reshape(np.array(state_lst), (-1, self.state_dim))
            reward_lst = np.reshape(np.array(reward_lst), (-1, 1))
            next_state_lst = np.reshape(np.array(next_state_lst), (-1, self.state_dim))
            action_lst = np.reshape(np.array(action_lst), (-1, self.action_dim))
            action_prob_lst = np.reshape(np.array(action_prob_lst), (-1, 1))
            done_lst = np.reshape(np.array(done_lst), (-1, ))
            # print(np.mean(action_lst), end = ' ')

            # compute advantage function with GAE
            advantage_lst = self.compute_advantage(state_lst, reward_lst, next_state_lst, done_lst)
            # print(advantage_lst)
            # train both the agent and the value net
            # print("reward_lst : %s" % str(reward_lst))
            for _ in range(self.K):
                # train agent
                summary, _ = self.sess.run([self.merged, self.train_agent_op], feed_dict={
                    self.input_state_ph : state_lst,
                    self.action_prob_old_ph: action_prob_lst,
                    self.action_ph: action_lst,
                    self.advantage_ph: advantage_lst
                })
                self.saver.add_summary(summary)

                # train value net
                self.sess.run(self.train_value_op, feed_dict={
                    self.input_state_gae_ph : state_lst,
                    self.input_state_gae_target_ph : next_state_lst,
                    self.reward_ph : reward_lst
                })
            self.sess.run(self.replacement_value_op)
            return_lst.append(np.sum(reward_lst))

        # print(len(return_lst))
        self.env.close()
        self.explore_eps = max(self.explore_eps * self.explore_eps_decay, self.explore_eps_min)
        try:
            sum_ret = np.sum(return_lst)
            # sum_ret > 100
            return sum_ret, self.explore_eps
        except Exception as e:
            print(return_lst)
            print(len(return_lst))
            print(e)
        
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
        assert advantage.shape == (episode_len, 1)

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

    def learn(self, iters = 1500):
        '''
            for each iter 1-N
                (you can also try A3C parallel) multi-workers
                collect T transitions

                train K epochs(thanks to the clipped objective)
        '''
        ret_lst = []
        print_interval = 20
        test_threshold = 200
        save_threshold = 200
        
        cur_eps = 100
        for i in range(iters):
            self.iter = i
            render = False    
            ret, eps = self.train_one_step(render= render)
            cur_eps = eps

            ret_lst.append(ret)
            print("\riter: %d, ret: %.3f, eps: %.3f" % (i, ret, eps), end = '')
            
            # print avg ret and test
            if i % print_interval == 0 and i is not 0:
                avg_ret = np.mean(ret_lst)
                print(", avg ret: %.3f" % avg_ret)
                if avg_ret > test_threshold:
                    self.test()
                    test_threshold = avg_ret
                ret_lst.clear()
            
                # save model
                if avg_ret > save_threshold:
                    model_dir = self.env_name
                    if os.path.exists(model_dir) is False:
                        os.mkdir(model_dir)
                    model_name = os.path.join(model_dir, str(avg_ret))
                    self.save(model_name)
                    save_threshold = avg_ret

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
    agent = PPOAgent()
    # agent.load("CartPole-v1/22.523809523809526")
    agent.learn()