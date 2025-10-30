import numpy as np;
import math

class Bandit:

    def __init__(self, k):
        self.qstar = np.zeros(k)
        self.n = np.zeros(k)
        self.q = np.zeros(k)
        for i in range(k):
            self.qstar[i] = np.random.normal(scale=math.sqrt(1))
        self.optimal = np.argmax(self.qstar)
    
    def pull_lever(self, lever):
        reward = np.random.normal(loc=self.qstar[lever], scale=math.sqrt(1))
        return reward
    
    def choose_action(self,epsilon):
        rnd = np.random.rand()
        if rnd < epsilon:
            return np.random.randint(0, self.qstar.size)
        else:
            return np.argmax(self.q)
    
    def step(self, epsilon):
        a = self.choose_action(epsilon)
        r = self.pull_lever(a)
        self.n[a] = self.n[a] + 1
        self.q[a] = self.q[a] + (1 / self.n[a]) * (r - self.q[a])
        optimal = 0
        if a == self.optimal:
            optimal = 1
        return {
            'r': r,
            'optimal': optimal,
        }
    
    def reset(self):
        self.n = np.zeros(self.n.size)
        self.q = np.zeros(self.q.size)