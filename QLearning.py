import gymnasium as gym
import random
import numpy as np
env = gym.make("FrozenLake-v1")
alpha = 0.9
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.001
num_episodes = 10000
max_states = 100
q_table = np.zeros((env.observation_space.n,env.action_space.n))
def choose_action(state):
    if random.uniform(0,1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state,:])
for episode in range(num_episodes):
    state,_=env.reset()
    terminated = False
    for step in range(max_states):
        action = choose_action(state)
        next_state,reward,terminated,truncated,_ = env.step(action)
        old_value = q_table[state,action]
        next_max = np.max(q_table[next_state,:])
        q_table[state,action] = q_table[state,action] + alpha*(reward+gamma*next_max-q_table[state, action])
        state = next_state
        if terminated or truncated:
            break
    epsilon = max(min_epsilon,epsilon*epsilon_decay)
env = gym.make("FrozenLake-v1",render_mode='human')
for episode in range(6):
    state,_ = env.reset()
    terminated = False
    print("Episode",episode)
    for step in range(max_states):
        env.render()
        action = np.argmax(q_table[state,:])
        next_state,reward,terminated,truncated,_=env.step(action)
        state = next_state
        if terminated or truncated:
            env.render()
            print("Episode:",episode,",Reward:",reward)
            break
env.close()