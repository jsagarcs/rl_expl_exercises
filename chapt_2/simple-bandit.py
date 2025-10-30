import numpy as np;
import math
import matplotlib.pyplot as plt

class Bandit:

    def __init__(self, k):
        self.qstar = np.zeros(k)
        self.n = np.zeros(k)
        self.q = np.zeros(k)
        for i in range(k):
            self.qstar[i] = np.random.normal(scale=math.sqrt(0))
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

class Learning_Method:
    def __init__(self, epsilon, num_steps):
        self.epsilon = epsilon
        self.reward_history = np.zeros(num_steps)
        self.optimal_action_history = np.zeros(num_steps)

k = 10
num_bandits = 2000
num_steps = 1000
bandits = []

methods = [
    Learning_Method(0, num_steps),
    Learning_Method(0.1, num_steps),
    Learning_Method(0.01, num_steps),
]

for i in range(num_bandits):
    bandits.append(Bandit(k))

for method in methods:
    for bandit_index, bandit in enumerate(bandits):
        for step in range(num_steps):
            results = bandit.step(method.epsilon)
            current_reward_total = method.reward_history[step]
            method.reward_history[step] = current_reward_total + results['r']
            current_optimal_action_total = method.optimal_action_history[step]
            method.optimal_action_history[step] = current_optimal_action_total + results['optimal']
        bandit.reset()
    method.reward_history = list(map(lambda x: x / num_bandits, method.reward_history))
    method.optimal_action_history = list(map(lambda x: x * (100 / num_bandits), method.optimal_action_history))
    print("method " + str(method.epsilon) + " done")

x = np.arange(1, num_steps + 1)
fig, (reward_graph, optimum_graph) = plt.subplots(2,1)
for method in methods:
    reward_graph.plot(x, method.reward_history, label=method.epsilon)
    optimum_graph.plot(x, method.optimal_action_history, label=method.epsilon)
plt.legend()
# print("Reward average")
# print(methods[0].reward_history)
# print("Optimal average")
# print(methods[0].optimal_action_history)
plt.savefig('./simple_bandit.png')
plt.show()