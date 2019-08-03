import tensorflow as tf
import numpy as np
import gym
import os
import pickle
'''
    Proximal Policy Optimization(PPO)算法在离散行为空间下的实现
    详细细节请参考笔记"PPO 推导"
    [这次不用actor-critic style, 而使用GAE]
    1. 网络结构:
        agent网络, 输入state, 输出action的分布，2个FC层
    2. 训练过程:
        loss = clipped surrogate objective, 见笔记
             = min(r * A, clip(r, 1-eps, 1+eps) * A)
    3. advantage function实现方法:
        使用GAE实现. A(s, a) = \sum_t (\gamma * \lambda)^t \delta_t
            \delta_t = r_t + \gamma * V(s_{t+1}) - V(s_t)
        里面的V使用神经网络输出，训练方法类似DQN，详情见笔记。
    4. 问题修复:
        1. 最开始的一组参数有的时候work，有的时候不work: 解决办法: K = 10, lra = 1e-4, lrc=2e-4
        2. 输出爆炸: 因为除法中的分母可能是0: 需要设置一个下界的1e-5
'''

class PPOAgent:
    def __init__(self):
        # init paras
        # 这样的参数有的时候会成功，有的时候会失败
        self.para_operation(mode="init")

        # create env
        self.env = self._create_env(self.env_name)
        self.action_dim = self.env.action_space.n
        self.state_dim = self.env.observation_space.shape[0]
        
        # build net
        self._build_network()

    def para_operation(self, mode = "init", name = None):
        param_num = 12
        if mode == "init" and name is None:
            self.env_name = "CartPole-v1"
            self.horizon_T = 40 # timestep segements between training
            self.K = 10         # K epochs in each iter (PPO specified)
            self.eps = 0.2      # the clip epsilon in surrogate objective
            self.lr_a = 0.0001  # learning rate for action net
            self.lr_v = 0.0002  # learning rate for value net
            self.tau = 0.002    # target net and eval net soft update speed
            self.gamma = 0.999   # reward -> return decay
            self.explore_eps = 5.0 # if rand() < explore_eps, take an random action
            self.explore_eps_decay = 0.99 # decay
            self.explore_eps_min = 0.05 # min threshold
            self.lmbda = 0.9    # lambda in GAE

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

    def _build_network(self):
        assert self.state_dim > 0 and self.action_dim > 0 
        # main agent policy network
        with tf.variable_scope("policy_net"):
            self.input_state_ph = tf.placeholder(dtype = tf.float32, shape = [None, self.state_dim], name = "input_state")
            l1 = tf.layers.dense(inputs = self.input_state_ph, units = 32, activation = tf.nn.relu, name = "l1")
            l2 = tf.layers.dense(inputs = l1,units = 64, activation = tf.nn.relu, name = "l2")
            self.output_action_prob = tf.nn.softmax(tf.layers.dense(inputs = l2, units = self.action_dim, activation=None, name = "output_action_prob"))
            # 如何训练agent? 如何训练gae
            # agent: 构建clipped surrogate objective
            self.action_prob_old_ph = tf.placeholder(dtype = tf.float32, shape = None, name = "action_prob_old_ph")
            self.action_ph = tf.placeholder(dtype = tf.int32, shape = None, name = "action_ph")
            # 制作mask, 正向传播和mask相乘后squeeze，然后和action_prob_old_ph做element wise的除法，得到ratio
            action_mask = tf.one_hot(indices = self.action_ph, depth = self.action_dim, name = "action_mask")
            action_prob_cur = tf.reduce_sum(action_mask * self.output_action_prob, axis = 1)
            ratio = tf.div(action_prob_cur, tf.maximum(self.action_prob_old_ph, 1e-5))
            self.advantage_ph = tf.placeholder(dtype = tf.float32, shape = None, name = "advantage_ph")

            # 计算surr1 和 surr2，取min得到最终loss。从而得到train op
            surr1 = ratio * self.advantage_ph
            surr2 = tf.clip_by_value(ratio, clip_value_min = 1 - self.eps, clip_value_max = 1+ self.eps) * self.advantage_ph
            # loss = tf.squeeze(-tf.minimum(surr1, surr2, name = "clipped_obj"))
            loss = -tf.minimum(surr1, surr2, name = "clipped_obj")
            self.train_agent_op = tf.train.AdamOptimizer(self.lr_a).minimize(loss)

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
            self.reward_ph = tf.placeholder(dtype = tf.float32, shape = None, name = "reward_ph")

            loss = tf.reduce_mean(self.reward_ph + self.gamma * self.output_value_gae_target - self.output_value_gae_eval)
            self.train_value_op = tf.train.AdamOptimizer(self.lr_v).minimize(loss)
            self.replacement_value_op = replacement

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _create_env(self, env_name):
        env = gym.make(env_name)
        env.reset()
        return env

    def get_action(self, state, test = False):
        assert state.shape == (1, self.state_dim)

        action_prob_tensor = self.sess.run(self.output_action_prob, feed_dict={
            self.input_state_ph : state
        })[0]
        if test is False and np.random.rand() < self.explore_eps:
            action = np.random.randint(self.action_dim)
        else:
            # print(state)
            # print(action_prob_tensor)
            action = np.random.choice(self.action_dim, p = action_prob_tensor)
        # print(action)
        # print(action_prob_tensor)
        action_prob = action_prob_tensor[action]
        
        action = int(action)
        assert 0 <= action < self.action_dim
        return action, action_prob
        
    def train_one_step(self):
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
            reward_lst = np.reshape(np.array(reward_lst), (-1, ))
            next_state_lst = np.reshape(np.array(next_state_lst), (-1, self.state_dim))
            action_lst = np.reshape(np.array(action_lst), (-1, ))
            action_prob_lst = np.reshape(np.array(action_prob_lst), (-1, ))
            done_lst = np.reshape(np.array(done_lst), (-1, ))

            # compute advantage function with GAE
            advantage_lst = self.compute_advantage(state_lst, reward_lst, next_state_lst, done_lst)

            # train both the agent and the value net
            for _ in range(self.K):
                # train agent
                self.sess.run(self.train_agent_op, feed_dict={
                    self.input_state_ph : state_lst,
                    self.action_prob_old_ph: action_prob_lst,
                    self.action_ph: action_lst,
                    self.advantage_ph: advantage_lst
                })

                # train value net
                self.sess.run(self.train_value_op, feed_dict={
                    self.input_state_gae_ph : state_lst,
                    self.input_state_gae_target_ph : next_state_lst,
                    self.reward_ph : reward_lst
                })
            self.sess.run(self.replacement_value_op)
            return_lst.append(np.sum(reward_lst))

        self.explore_eps = max(self.explore_eps * self.explore_eps_decay, self.explore_eps_min)
        try:
            sum_ret = np.sum(return_lst)
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
        assert reward.shape == (episode_len, )
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
        advantage = np.reshape(np.array(advantage_lst), (-1, ))
        assert advantage.shape == (episode_len, )

        return advantage
            
    def test(self):
        state = self.env.reset()
        while True:
            self.env.render()
            action, _ = self.get_action(np.reshape(state, (1, self.state_dim)), test = True)
            state, reward, done , _ = self.env.step(action)
            if done == True:
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
        for i in range(iters):
            ret, eps = self.train_one_step()
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
    print("succ")
    agent = PPOAgent()
    # agent.load("CartPole-v1/22.523809523809526")
    agent.learn()