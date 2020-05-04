import copy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(object):
    """Interacts with and learns from the environment."""
    def __init__(self, actor_local, actor_target, critic_local, critic_target, noise, memory):
        """Initialize an Agent object.
        
        Args:
            actor_local (torch.nn.Module): local actor model.

            actor_target (torch.nn.Module): target actor model.

            critic_local (torch.nn.Module): local critic model.

            critic_target (torch.nn.Module): target critic model.

            noise (OUNoise): a noise process.

            memory (ReplayBuffer): a ReplayBuffer for experience storage.
        """
        # Actor Network (w/ Target Network)
        self.actor_local = actor_local
        self.actor_target = actor_target
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = critic_local
        self.critic_target = critic_target
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = noise

        # Replay memory
        self.memory = memory

    def reset(self):
        """Reset internal state of noise process."""
        self.noise.reset()

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy.

        Args:
            state (array): present environment state (batch).

            add_noise (boolean): flag, whether noise shall be considered (defaults to True).

        Returns:
            *array*: action associated with state (batch).
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        """Make the model learn from one sample of replay buffer."""
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self._learn(experiences)

    def _learn(self, experiences):
        """Update actor (policy) & critic (value) model parameters using given batch of experiences.

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        """
        self._update_critic_network(experiences)
        self._update_actor_network(experiences)
        self._update_target_networks()

    def _update_critic_network(self, experiences):
        """Updates the critic (value) network from a batch of experiences.

        Note:
            * Q_targets = r + γ * critic_target(next_state, actor_target(next_state))

            where:

                actor_target(state) -> action

                critic_target(state, action) -> Q-value

            * Q_expected = critic_local(states, actions)

            * Loss_critic = MSE(Q_targets, Q_expected)

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples.
        """
        states, actions, rewards, next_states, dones = experiences
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        # Compute and minimize critic loss
        Q_expected = self.critic_local(states, actions)
        loss_critic = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

    def _update_actor_network(self, experiences):
        """Updates the actor (policy) network from a batch of experiences.

        Note:
            Loss_actor = -mean(critic_local(states, actor(states)))

        Args:
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples.
        """
        states, actions, rewards, next_states, dones = experiences
        actions_pred = self.actor_local(states)
        loss_actor = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()

    def _update_target_networks(self):
        self._soft_update(self.critic_local, self.critic_target, TAU)
        self._soft_update(self.actor_local, self.actor_target, TAU)

    def _soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.

        Note:
            θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model: PyTorch model (weights will be copied from)

            target_model: PyTorch model (weights will be copied to)

            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.05):
        """Initialize parameters and noise process.

        Args:
            size: size of the random variable (noise).

            seed: random seed.

            mu: mean reversion level (equilibrium position).

            theta: mean reversion rate (rigidity of Ornstein-Uhlenbeck process).

            sigma: diffusion (impact of randomness on outcome of the process).
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.rand(len(x))
        self.state = x + dx
        return self.state
