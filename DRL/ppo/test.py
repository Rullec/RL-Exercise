import gym
import torch
import torch.nn as nn   # 这个nn就是全连接层
import torch.nn.functional as F
import torch.optim as optim # 应该是优化器
from torch.distributions import Categorical


# hyper paras
lr = 0.0005
gamma = 0.98    # decay between value function
lmbda = 0.95    # used in GAE
eps_clip = 0.1
K_epoch = 3
T_horizon = 20  # 这个不知道是干啥的...

class PPO(nn.Module):
    def __init__(self):
        # 调用父类构造函数init
        super(PPO, self).__init__()

        # 推测是buffer
        self.data = []

        self.fc1 = nn.Linear(4, 256)    # 似乎输入是4, 输出是256
        self.fc_pi = nn.Linear(256, 2)  # 叫pi的全连接？
        self.fc_v= nn.Linear(256, 1)    # 这个输出是1,可能是值函数

        # self.parameters 返回一个在module参数上的迭代器
        self.optimizer = optim.Adam(self.parameters(), lr = lr) 

    def pi(self, x, softmax_dim = 0):
        '''
            一个relu的fc -> 一个fc -> 一个softmax
        '''
        x = F.relu(self.fc1(x)) # 加了一个fc, 然后relu
        x = self.fc_pi(x)   # x
        prob = F.softmax(x, dim = softmax_dim)
        return prob
    
    def v(self, x):
        '''
            relu的fc -> fc 
            推测这个是值函数, fc_v是值函数
        '''
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        # 这个应该是buffer
        self.data.append(transition)

    def make_batch(self):
        # 制作batch
        s_lst, a_lst, r_lst, \
        s_prime_lst, prob_a_lst, done_lst = \
            [], [], [], [], [], []
        for transition in self.data:
            # state, action, reward, state_next, p(a|s), done
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
    
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)

        # 缓存清空
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
    
    def train_net(self):
        
        # get data, prob_a is a list of prob of exceuted actions p(a|s) in these transitions
        s, a, r, s_prime, done_mask, prob_a = \
            self.make_batch()
        
        # train K_epoch
        for i in range(K_epoch):
            # td error = r + gamma * V(s_next) - V(s)
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0

            # Generalized Advantage Estimator
            # GAE: advantage[i] = gamma * lambda * advantage[i+1] + TD_error
            # 倒序访问delta
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype = torch.float)

            pi = self.pi(s, softmax_dim = 1)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a) - log(b))

            # 实现clip surrogate objective
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            # zero_grad 将所有梯度设置为0
            self.optimizer.zero_grad()
            loss.mean().backward()  # mean或者sum都可以
            self.optimizer.step()   # 执行一步优化
        
    
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    model = PPO()
    score = 0.0
    print_interval = 20

    for n_epi in range(500):
        s = env.reset()
        done = False

        # episode = 500, 一共进行过500个episode
        while not done:

            # horizon进行20个step
            for t in range(T_horizon):
                # 获得discrete prob distribution的分布
                prob = model.pi(torch.from_numpy(s).float())
                
                # 离散action space, 拿到概率
                m = Categorical(prob)  
                
                # 采样获得action
                a = m.sample().item()

                # 执行action
                s_prime, r, done, info = env.step(a)

                # 把元组存入buffer中
                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                s = s_prime

                score += r
                if done:
                    break
            
            # 训练网络,
            model.train_net()


        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            def test():
                s = env.reset()
                while True:
                    env.render()
                    prob = model.pi(torch.from_numpy(s).float())
                    m = Categorical(prob)
                    a = m.sample().item()
                    s, r, done, info = env.step(a)
                    if done == True:
                        env.close()
                        break

            if score / print_interval > 200:
                test()
            score = 0.0

    env.close()