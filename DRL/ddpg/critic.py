import tensorflow as tf
import numpy as np

class Critic:
    '''

    '''
    def __init__(self, units, state_dims, action_dims, lr, gamma, replacement_dict, sess):
        # init var
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.lr = lr
        self.gamma = gamma
        self.sess = sess

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
        self.eval_output = self._build_single_net(self.input_s, self.input_a, scope_name = "eval_net",
            trainable = True)
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "Critic/eval_net")

        # create target net(target 网络，不可训练s)
        self.targe_output = self._build_single_net(self.input_s, self.input_a, scope_name = "target_net",
            trainable = False)
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "Critic/target_net")

        # 定义loss: \reward + \gamma * Q(s_, \mu(s_)) - Q(s_a)
        self.q_next = tf.placeholder(dtype = tf.float32, shape = (None, ), name = "q_next")
        self.reward_ph = tf.placeholder(dtype = tf.float32, shape = (None, ), name = "reward_ph")
        self.loss = tf.reduce_mean(tf.squared_difference(self.reward_ph + self.gamma * self.q_next, self.eval_output))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # 由于actor网络的训练需要dq/da，也就是让critic主网络求一下梯度...这个操作也需要在build network这里定义一个op
        self.dqda = tf.gradients(ys = self.eval_output, xs = self.input_a)

        # 定义target replacement
        # 定义参数replace过程(就是用actor参数覆盖target参数)
        self.para_replacement = []
        if self.update_target_way == "hard":
            if self.update_target_iter_cur % self.update_target_iter == 0:
                self.para_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]

        elif self.update_target_way == "soft":
            self.para_replacement = [tf.assign(t,
                (1 - self.update_target_tau) * t + self.update_target_tau * e)
                for t, e in zip(self.t_params, self.e_params)]
        else:
            assert ValueError, "the update way is illegal"
                
    def _build_single_net(self, input_s, input_a, scope_name, trainable):
        init_w = tf.random_normal_initializer(0.0, 0.3)
        init_b = tf.constant_initializer(0.1)
        with tf.variable_scope(scope_name):
            with tf.variable_scope("l1"):
                units = 32
                w1_s = tf.get_variable("w1_s", shape = [self.state_dims, units], 
                initializer = init_w, trainable=trainable)
                w1_a = tf.get_variable("w1_a", shape = [self.action_dims, units], 
                initializer = init_w, trainable=trainable)
                b1 = tf.get_variable("b1", shape = [1, units], initializer= init_b,\
                    trainable = trainable)
                l1 = tf.nn.relu(tf.add(tf.matmul(input_s, w1_s) + tf.matmul(input_a, w1_a), b1))
            
            with tf.variable_scope("l2"):
                l2 = tf.layers.dense(name="l2", units = 64,inputs = l1,
                trainable = trainable, activation = tf.nn.relu, kernel_initializer = init_w,
                 bias_initializer = init_b )

            with tf.variable_scope("q"):
                output = tf.layers.dense(name = "output", units = 1, inputs = l2,
                    trainable = trainable, activation=None, kernel_initializer = init_w,
                    bias_initializer = init_b)
        
        return output

    def train(self, state, action, reward, q_next):
        '''
            这个函数除了训练以外，还负责计算dqda的梯度，送到外面去，给actor训练用
        '''
        assert state.shape[1] == self.state_dims
        batch_size = state.shape[0]
        assert action.shape == (batch_size, self.action_dims)
        assert reward.shape == (batch_size, )
        assert q_next.shape == (batch_size, )

        
        # 训练网络, 获得dqda的梯度
        _, batch_dq_da = self.sess.run([self.train_op, self.dqda], feed_dict = {
            self.input_a: action,
            self.input_s: state,
            self.reward_ph: reward,
            self.q_next: q_next
        })
        batch_dq_da = np.squeeze(np.array(batch_dq_da), axis=0)

        # 更新参数
        if self.update_target_way == "hard":
            self.update_target_iter_cur += 1
            if self.update_target_iter_cur % self.update_target_iter == 0:
                self.sess.run(self.para_replacement)
        elif self.update_target_way == "soft":
            self.sess.run(self.para_replacement)
        else:
            assert 0 == 1, "replacement illegal"

        assert batch_dq_da.shape == (batch_size, self.action_dims)
        return batch_dq_da

    def get_target_q(self, state, action):
        '''
            state: (X, self.state_dims)
            action: (X, self.action_dims)
        '''
        assert state.shape[1] == self.state_dims
        assert action.shape[1] == self.action_dims
        assert state.shape[0] == action.shape[0]

        target_q = self.sess.run(self.targe_output, feed_dict = {
            self.input_a: action,
            self.input_s : state
        })
        target_q = np.reshape(target_q, [state.shape[0]])

        assert target_q.shape == (state.shape[0], )
        return target_q

    def get_dq_da(self, state, action):
        '''
            求取梯度，用于actor的policy gradient计算
            state: (X, state_dims) 
            action: (X, action_dims)
            dqda: (X, action_dims)
        '''
        assert state.shape[1] == self.state_dims
        assert action.shape[1] == self.action_dims
        assert state.shape[0] == self.action_dims.shape[0]

        dqda = self.sess.run(self.dqda, feed_dict = {
            self.input_a: action,
            self.input_s: state
        })

        assert dqda.shape == (state.shape[0], self.action_dims)
        return dqda