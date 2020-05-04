from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from src.ddpg_agent import Agent
from src.ddpg_agent import OUNoise
from src.model import Actor
from src.model import Critic
from src.model import load_actor
from src.model import load_critic
from src.replay_buffer import ReplayBuffer

####################################################################################################


# select this option to load version 1 (with a single agent) of the environment
# env = UnityEnvironment(file_name='/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')
# select this option to load version 2 (with 20 agents) of the environment
# env = UnityEnvironment(file_name='/data/Reacher_Linux_NoVis/Reacher.x86_64')
env = UnityEnvironment(file_name='resources/environments/Reacher_Windows_x86_64_twenty/Reacher.exe')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


####################################################################################################


BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
random_seed = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Actor Network (w/ Target Network)
actor_local = Actor(state_size, action_size, random_seed).to(device)
actor_target = Actor(state_size, action_size, random_seed).to(device)

# Critic Network (w/ Target Network)
critic_local = Critic(state_size+action_size, 1, random_seed).to(device)
critic_target = Critic(state_size+action_size, 1, random_seed).to(device)

# Noise process
noise_process = OUNoise(action_size, random_seed)

# Replay memory
memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed, device)


####################################################################################################

# train the Agent
max_episodes = 300
max_timesteps = 1000

def ddpg(n_episodes=max_episodes, n_timesteps=max_timesteps):
    """Deep Deterministic Policy Gradient.

    Args:
        n_episodes (int): maximum number of training episodes
        n_timesteps (int): maximum number of timesteps per episode
    """
    mean_scores = []  # list containing average scores from each episode
    mean_scores_window = deque(maxlen=100)  # last 100 average scores

    agent = Agent(actor_local, actor_target, critic_local, critic_target, noise_process, memory)

    for episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        # initialize the mean score (averaged over each agent)
        scores = np.zeros(num_agents)
        mean_score = 0
        for timestep in range(n_timesteps):
            actions = agent.act(states)
            actions = np.clip(actions, -1, 1)
            # send all actions to the environment
            env_info = env.step(actions)[brain_name]
            # get next state (for each agent)
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            for i in range(num_agents):
                agent.step(states[i], actions[i], rewards[i], next_states[i], dones[i])
            agent.learn()
            states = next_states
            scores += rewards
            if timestep % 10 == 0:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                print('\rTimestep {}\tMean Score: {:.2f}\tStd Score: {:.2f}'.
                      format(timestep, mean_score, std_score), end="")
            if np.any(dones):
                break

        mean_scores_window.append(mean_score)  # save most recent score
        mean_scores.append(mean_score)  # save most recent score

        if episode % 1 == 0:
            print(
                '\nEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(mean_scores_window)))
        if np.mean(mean_scores_window) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                episode - 100, np.mean(mean_scores_window)))
            agent.actor_local.save_checkpoint("checkpoint_actor.pth")
            agent.actor_local.save_model("model_actor.pt")
            agent.critic_local.save_checkpoint("checkpoint_critic.pth")
            agent.critic_local.save_model("model_critic.pt")
            break
    return mean_scores


avg_scores = ddpg()


####################################################################################################


# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(avg_scores)), avg_scores)
plt.ylabel('Average Score')
plt.xlabel('Episode #')
plt.title('DDPG Agent results: Average Scores vs. Episode #')
plt.show()


####################################################################################################


agent = Agent(actor_local, actor_target, critic_local, critic_target, noise_process, memory)
agent.actor_local = load_actor('resources/models/model_actor.pt')
agent.critic_local = load_critic('resources/models/model_critic.pt')

for i_episode in range(1):
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations
    scores = 0
    while True:
        actions = agent.act(states)
        actions = np.clip(actions, -1, 1)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        states = next_states
        scores += np.mean(rewards)
        if any(dones):
            break
    print("Episode: {}, Score: {}".format(i_episode, scores))

env.close()
