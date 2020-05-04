import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_actor(filename, seed=0):
    """Loads a model from file and returns a Actor model object.

    Args:
        filename (str): actor model filename to be loaded.

        seed (int): random seed.

    Returns:
        torch.nn.Module: actor model.
    """
    checkpoint = torch.load(filename)
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    hidden_sizes = checkpoint['hidden_sizes']
    actor = Actor(input_size, output_size, seed, *hidden_sizes)
    actor.load_state_dict(checkpoint['state_dict'])
    return actor


def load_critic(filename, seed=0):
    """Loads a model from file and returns a Critic model object.

    Args:
        filename (str): critic model filename to be loaded.

        seed (int): random seed.

    Returns:
        torch.nn.Module: critic model.
    """
    checkpoint = torch.load(filename)
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    hidden_sizes = checkpoint['hidden_sizes']
    critic = Critic(input_size, output_size, seed, *hidden_sizes)
    critic.load_state_dict(checkpoint['state_dict'])
    return critic


def hidden_init(layer):
    """Returns the uniform distribution range for hidden layer weight initialization.

    Args:
        layer: model hidden layer.

    Returns:
        tuple: uniform distribution range for hidden layer weight initialization.
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Deterministic Policy) Model."""
    def __init__(self, input_size, output_size, seed, fc1_units=40, fc2_units=20):
        """Initialize parameters and build model.

        Args:
            input_size (int): model input dimension (here: state_size).

            output_size (int): model output dimension (here: action_size).

            seed (int): Random seed.

            fc1_units (int): Number of nodes in first hidden layer (defaults to 40).

            fc2_units (int): Number of nodes in second hidden layer (defaults to 20).
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, output_size)
        self.reset_parameters()

    def reset_parameters(self):
        """Randomly initialize hidden layer weights as drawn from a uniform distribution."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions.

        Args:
            state (tensor): state of environment (batch).

        Returns:
            action (tensor): action to be applied to environment (batch).
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))

    def save_model(self, filename='model_actor.pt'):
        """Saves model to file (architecture + state_dict (weights)).

        Args:
            filename (str): model filename (defaults to "model_actor.pt")
        """
        checkpoint = {
            'input_size': self.fc1.in_features,
            'output_size': self.fc3.out_features,
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
    def __init__(self, input_size, output_size, seed, fc1_units=40, fc2_units=20):
        """Initialize parameters and build model.

        Args:
            input_size (int): model input dimension (here: state_size + action_size).

            output_size (int): model output dimension (here: constant = 1).

            seed (int): Random seed.

            fc1_units (int): Number of nodes in the first hidden layer (defaults to 40).

            fc2_units (int): Number of nodes in the second hidden layer (defaults to 40).
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, output_size)
        self.reset_parameters()

    def reset_parameters(self):
        """Randomly initialize hidden layer weights as drawn from a uniform distribution."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values.

        Args:
            state (tensor): state of environment (batch).

            action (tensor): action to be applied in environment (batch).

        Returns:
            tensor: state + action values (batch).
        """
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
            'input_size': self.fc1.in_features,
            'output_size': self.fc3.out_features,
            'hidden_sizes': [layer.out_features for layer in [self.fc1, self.fc2]],
            'state_dict': self.state_dict()}
        torch.save(checkpoint, filename)

    def save_checkpoint(self, filename='checkpoint_critic.pth'):
        """Saves model weights to file only.

        Args:
            filename (str): checkpoint filename (defaults to "checkpoint_critic.pt")
        """
        torch.save(self.state_dict(), filename)
