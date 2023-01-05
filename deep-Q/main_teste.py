# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 01:34:59 2023

@author: neiju
"""

import gym
from Utils.brain import Dqn
import numpy as np

env = gym.make('MountainCar-v0', render_mode="rgb_array")

print('State space: ', env.observation_space)
print(env.observation_space.low)
print(env.observation_space.high)

print('State space: ', env.action_space)

state = env.reset()
reward = 0

brain = Dqn(2,3,0.9)

#print(np.array([state[0]]), np.float32)
action = brain.update(state,reward)
print(state)
action = brain.update(state,reward)
print(action)

next_state, reward, done, info, _ = env.step(action)

print("state: ", state)
next_state = np.array((np.array(next_state, dtype = 'f')))
next_state = next_state,{}
print("next_state: ",(next_state,{} ))

action = brain.update(next_state,reward)


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
        action = self.model.update(state,reward)
        tot_reward = 0
        max_x = -100
        self.list_steps = []
        print("funcionou 1")
        teste = 1;
        while True:
            teste = teste + 1;
            print(teste)
            if self.render:
                self.env.render()
            print("funcionou 2")
            self.steps += 1
            next_state, reward, done, info, _ = env.step(action)
            print("funcionou 3")
            if next_state[0] >= -0.25:
                reward += 1  
            elif next_state[0] >= 0.1:
                reward += 1
            elif next_state[0] >= 0.25:
                reward += 1  
            elif next_state[0] >= 0.5:
                reward += 200
            print("funcionou 4")
            next_state = np.array((np.array(next_state, dtype = 'f')))
            next_state = next_state,{}
            action = self.model.update(next_state,reward)
            print("funcionou 5")
            print(next_state[0][0])
            if next_state[0][0] > max_x:
                max_x = next_state[0][0]
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
                print("Step {}, Total reward: {}, Max: ".format(self.steps, tot_reward,max_x))
                if self.steps < 180:
                    self.model.save()
                self.steps = 0
                break
            
steps = [];
gr = GameRunner(model = brain,env = env,render=True)
for i in range(20000):
    gr.run()
