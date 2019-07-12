import numpy as np
import tensorflow as tf

class Actor:
    def __init__(self, state_dims, action_dims,\
         replacement_dict, action_low_bound, action_high_bound,\
              lr, max_explore_iter, sess):
        
        # importane variables
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.lr = lr
        self.max_explore_iter = max_explore_iter
        self.cur_explore_iter = 0
        self.action_low_bound = action_low_bound    # the action is a real number, the upper bound is XX
        self.action_high_bound = action_high_bound        # and the lower bound is XX
        self.sess = sess

        # replacement policy
        self.update_target_way = replacement_dict["name"]
        if self.update_target_way == "hard":
            self.update_target_iter = replacement_dict["rep_iter_actor"]
            self.update_target_iter_cur = 1
        elif replacement_dict["name"] == "soft":
            self.update_target_tau = replacement_dict["tau"]
        else:
            assert ValueError, "the policy is illegal"

        # build network
        self._build_network()

    def _build_network(self):
        '''
            actor网络:
            input: s, state_dims
            output: a , real number
            arch: 2 FC layers
        '''
        self.input = tf.placeholder(dtype = tf.float32, shape=(None, self.state_dims), name="input_state")
        self.dqda = tf.placeholder(dtype = tf.float32, shape = (None, self.action_dims), name ="input_dqda")

        # eval network
        self.eval_output = self._build_single_net(self.input, "eval_net", trainable = True)
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Actor/eval_net")
        
        # 定义loss
        with tf.variable_scope("policy_gradient"):
            self.policy_grads = tf.gradients(ys = self.eval_output, xs = self.e_params, grad_ys = -self.dqda)

        with tf.variable_scope("actor_train"):
            opt = tf.train.AdamOptimizer(self.lr)
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))
     
        # target network
        self.target_output = self._build_single_net(self.input, "target_net", trainable = False)
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Actor/target_net")

        # 定义参数replace过程(就是用actor参数覆盖target参数)
        self.para_replacement = [tf.assign(t, (1 - self.update_target_tau) * t + self.update_target_tau * e)
                                for t, e in zip(self.t_params, self.e_params)]

    def _build_single_net(self, input, scope_name, trainable):
        # input placeholder
        init_w = tf.random_normal_initializer(0.0, 0.5)
        init_b = tf.constant_initializer(0.1)

        with tf.variable_scope(scope_name):
            l1 = tf.layers.dense(inputs = input, 
                units = 32,
                activation = tf.nn.relu,
                kernel_initializer = init_w,
                bias_initializer = init_b,
                trainable = trainable,
                name = "l1",
                )
            l2 = tf.layers.dense(inputs = l1, 
                units = 64,
                activation = tf.nn.relu,
                kernel_initializer = init_w,
                bias_initializer = init_b,
                trainable = trainable,
                name = "l2",
                )
            before_output = tf.layers.dense(inputs = l2, 
                units = self.action_dims,
                activation = tf.nn.tanh,
                kernel_initializer = init_w,
                bias_initializer = init_b,
                trainable = trainable,
                name = "before_output")
            # 获得forward propogation的结果以后，网络结构完成
            output = tf.multiply(before_output, (self.action_high_bound - self.action_low_bound)/2.0 , name="output_action")
        return output
    
    def train(self, state, dqda):
        '''
            this function will train the actor network
            state : (-1, self.state_dims)
            dqda : (-1, self.action_dims)
        '''
        assert state.shape[0] == dqda.shape[0]
        assert state.shape[1] == self.state_dims
        assert dqda.shape[1] == self.action_dims

        # actor的训练
        self.cur_explore_iter += 1
        _ = self.sess.run(self.train_op, feed_dict = {
            self.input: state,
            self.dqda: dqda
        })
        
        # 更新replacement
        # saved1 = self.sess.run(self.t_params)
        self.sess.run(self.para_replacement)
        # saved2 = self.sess.run(self.t_params)
        # print(np.array(saved1) - np.array(saved2))
        
    def get_target_action(self, state):
        '''
            输入state, 要求输出targe action,也就是让target network forward propogation
            这个函数主要是被critic网络调用; 
            state: (X, state_dims)
            action: (X, action_dims)
        '''
        assert state.shape[1] == self.state_dims

        action = self.sess.run(self.target_output, feed_dict = {
            self.input : state
        })
        
        assert action.shape == (state.shape[0], self.action_dims)
        return action

    def get_action(self, state, test = False):
        '''
            get action from a state(s)

            state: (X, self.state_dims)
            action: (X, self.action_dims)

        '''
        # print(type(state))
        # print(state.shape)
        assert state.shape[1] == self.state_dims


        if self.cur_explore_iter / self.max_explore_iter < np.random.rand() and test == False:
            # 探索
            action = self.random_action()
        else:
            action = self.sess.run(self.eval_output, feed_dict ={
            self.input : state
            })
        
        assert action.shape == (state.shape[0], self.action_dims)
        return action
   
    def random_action(self, num = 1):
        '''
            action: (num, self.action_dims)
        '''
        action = np.random.rand(num, self.action_dims) * (self.action_high_bound - self.action_low_bound)
        action += self.action_low_bound

        assert action.shape == (num, self.action_dims)
        return action
    
    def get_epsilon(self):
        return (self.cur_explore_iter / self.max_explore_iter)