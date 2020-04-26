import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_actor(filename, seed=0):
    """Loads a model from file and returns a Actor model object."""
    checkpoint = torch.load(filename)
    state_size = checkpoint['state_size']
    action_size = checkpoint['action_size']
    hidden_sizes = checkpoint['hidden_sizes']
    actor = Actor(state_size, action_size, seed, *hidden_sizes)
    actor.load_state_dict(checkpoint['state_dict'])
    return actor


def load_critic(filename, seed=0):
    """Loads a model from file and returns a Critic model object."""
    checkpoint = torch.load(filename)
    state_action_size = checkpoint['state_action_size']
    value_size = checkpoint['value_size']
    hidden_sizes = checkpoint['hidden_sizes']
    actor = Critic(state_action_size, value_size, seed, *hidden_sizes)
    actor.load_state_dict(checkpoint['state_dict'])
    return actor


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Deterministic Policy) Model."""
    def __init__(self, state_size, action_size, seed, fc1_units=40, fc2_units=20):
        """Initialize parameters and build model.

        Args:
            state_size (int): Dimension of each state

            action_size (int): Dimension of each action

            seed (int): Random seed

            fc1_units (int): Number of nodes in first hidden layer

            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))

    def save_model(self, filename='model_actor.pt'):
        """Saves model to file (architecture + state_dict (weights)).

        Args:
            filename (str): model filename (defaults to "model_actor.pt")
        """
        checkpoint = {
            'state_size': self.fc1.in_features,
            'action_size': self.fc3.out_features,
            'hidden_sizes': [layer.out_features for layer in [self.fc1, self.fc2]],
            'state_dict': self.state_dict()}
        torch.save(checkpoint, filename)

    def save_checkpoint(self, filename='checkpoint_actor.pth'):
        """Saves model weights to file only.

        Args:
            filename (str): checkpoint filename (defaults to "checkpoint_actor.pt")
        """
        torch.save(self.state_dict(), filename)


class Critic(nn.Module):
    """Critic (Value) Model."""
    def __init__(self, state_size, action_size, seed, fc1_units=40, fc2_units=20):
        """Initialize parameters and build model.

        Args:
            state_size (int): Dimension of each state

            action_size (int): Dimension of each action

            seed (int): Random seed

            fc1_units (int): Number of nodes in the first hidden layer

            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size + action_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def save_model(self, filename='model_critic.pt'):
        """Saves model to file (architecture + state_dict (weights)).

        Args:
            filename (str): model filename (defaults to "model_critic.pt")
        """
        checkpoint = {
            'state_action_size': self.fc1.in_features,
            'value_size': self.fc3.out_features,
            'hidden_sizes': [layer.out_features for layer in [self.fc1, self.fc2]],
            'state_dict': self.state_dict()}
        torch.save(checkpoint, filename)

    def save_checkpoint(self, filename='checkpoint_critic.pth'):
        """Saves model weights to file only.

        Args:
            filename (str): checkpoint filename (defaults to "checkpoint_critic.pt")
        """
        torch.save(self.state_dict(), filename)
