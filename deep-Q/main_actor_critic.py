from Models.actor_critic_discrete import Agent

brain = Agent(2,2)

state = [1 , 2]
action = 2
reward = 0

brain.remember(state,action,reward)
brain.remember(state,action,reward)
brain.remember(state,action,reward)

state, action, reward = brain.replay()
print(reward[0].item())
