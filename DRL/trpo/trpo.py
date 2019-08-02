import tensorflow as tf
import numpy as np
import gym
import os
import pickle

class TRPOAgent:
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
        param_num = 1
        if mode == "init" and name is None:
            self.env_name = "CartPole-v1"

        elif mode == "load" and name is not None:
            config_path = name + ".conf"
            with open(config_path, "rb") as f:
                para = pickle.load(f)
                assert len(para) == param_num
                self.env_name = para["env_name"]
                print("load conf from %s succ: %s" % (config_path, str(para)))

        elif mode == "save" and name is not None:
            para = {"env_name": self.env_name}
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

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _create_env(self, env_name):
        env = gym.make(env_name)
        env.reset()
        return env

    def get_action(self, state, test = False):
        action = 0
        assert ValueError

        return action

    def train_one_step(self):
        '''
            run one iteration
        '''

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
        pass

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
        print("load %s succ" % name)
    
if __name__ =="__main__":
    agent = TRPOAgent()
    agent.learn()