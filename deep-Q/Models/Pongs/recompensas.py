import numpy as np
import torch

def discount_rewards(self, reward):
    # Compute the gamma-discounted rewards over an episode
    gamma = self.gamma    # discount rate
    running_add = 0
    discounted_r = np.zeros_like(reward)
    for i in reversed(range(0,len(reward))):
        if reward[i] != 0:
            running_add = 0 # (pong specific game boundary!)
        running_add = running_add * gamma + reward[i]
        discounted_r[i] = running_add

    discounted_r -= np.mean(discounted_r) # normalizing the result
    discounted_r /= np.std(discounted_r) # divide by standard deviation
    return torch.from_numpy(discounted_r)