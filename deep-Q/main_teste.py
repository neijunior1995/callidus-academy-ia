# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 01:34:59 2023

@author: neiju
"""

import gym
from Utils.brain import Dqn
import numpy as np

env = gym.make('MountainCar-v0', render_mode = "human")

print('State space: ', env.observation_space)
print(env.observation_space.low)
print(env.observation_space.high)

print('State space: ', env.action_space)

state = env.reset()
state = state[0].tolist()
print(state)
reward = 0

brain = Dqn(2,3,0.9)

print("Funcionou")
action = brain.update(state,reward)
print(action)

next_state, reward, done, info, _ = env.step(action)


print("-----------------")
print(next_state)
print(state)
print("-----------------")


action = brain.update(next_state,reward)
env.render()

class GameRunner:
    def __init__(self, model, env, render=True):
        self.env = env
        self.model = model
        self.render = render
        self.steps = 0
        self.list_steps = []
        self.reward_store = []
        self.max_x_store = []
    def run(self):
        state = self.env.reset()
        reward = 0
        state = state[0].tolist()
        action = self.model.update(state,reward)
        tot_reward = 0
        max_x = -100
        self.list_steps = []
        while True:
            self.steps = self.steps +1
            self.env.render()
            next_state, reward, done, info, _ = env.step(action)
            if next_state[0] >= -0.25:
                reward += 1  
            elif next_state[0] >= 0.1:
                reward += 1
            elif next_state[0] >= 0.25:
                reward += 1  
            elif next_state[0] >= 0.5:
                reward += 200
            action = self.model.update(next_state,reward)
            if next_state[0] > max_x:
                max_x = next_state[0]
                #print(max_x)
            if max_x > 0.5:
                print("You Win")
            # is the game complete? If so, set the next state to
            # None for storage sake
            if done or self.steps > 1000:
                next_state = None

            # move the agent to the next state and accumulate the reward
            state = next_state
            tot_reward += reward
            # if the game is done, break the loop
            if done or self.steps > 1000:
                self.reward_store.append(tot_reward)
                self.max_x_store.append(max_x)
                self.list_steps.append(self.steps)
                print("Step {}, Total reward: {}, Max {}: ".format(self.steps, tot_reward,max_x))
                if self.steps < 180:
                    self.model.save()
                self.steps = 0
                break
            
steps = [];
gr = GameRunner(model = brain,env = env,render=True)
for i in range(20000):
    gr.run()
