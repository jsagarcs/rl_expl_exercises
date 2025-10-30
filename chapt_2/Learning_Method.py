import numpy as np;

class Learning_Method:
    def __init__(self, epsilon, num_steps):
        self.epsilon = epsilon
        self.reward_history = np.zeros(num_steps)
        self.optimal_action_history = np.zeros(num_steps)