from Models.actor_critic_discrete import Agent
import torch

brain = Agent(2,2)

state = [1 , 2]
action = 2
reward = 0

brain.remember([1 , 2],0,0)
brain.remember([2,1],1,1)
brain.remember([2,3],0,0)

state, action, reward = brain.replay()
state = torch.Tensor(state[0]).float().unsqueeze(0)
new_action = brain.select_action(state)
brain.learn()
new_action = brain.select_action(state)
