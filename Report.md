# Report

This report discusses the points listed below in greater detail:
1.  Background on DQN
    1.  Q-Learning
    1.  Deep Learning
    1.  DQN-Algorithm
2.  Empirical Results
    2.  Reward-vs-Episodes-Plot
    2.  Gifs showing
        2.  the agent's performance before training 
        2.  the agent's performance after training        
3.  Implementation
    3.  Neural Network Architecture
    3.  Hyperparameters
4.  Possible Future Improvements
    4.  Double-DQN
    4.  Prioritised Experience replay
    4.  Duel-DQN
    4.  Rainbow

## Theoretical DQN Agent Design

The algorithm used is based on the DQN algorithm described in this paper by Deepmind back in 2015: 
https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

DQN (Deep-Q-Networks) is an approach in reinforcement learning that effectively combines two separate fields:

### Q-Learning

In classical reinforcement learning, the goal is to teach an agent to navigate a new environment with the goal of maximising cumulative rewards. One approach is Q-learning, where the agent tries to learn the dynamics of the environment by assigning values to <state, action>-pairs for the environment under test. This is achieved over the course of training, using it's experiences to produce and improve these estimates: As the agent encounters <state, action>-pairs more often, it becomes more and more "confident" of their value. In the end, an optimal policy can be obtained iteratively by starting from any state and always choosing that action that maximises the <state, action>-value for that state. 

### Deep Learning

Deep Learning is a part of a broader family of machine learning methods and is based upon artificial neural networks. The latter have frequently been used to obtain or even surpass human level performance. They lend themselves to solve tasks in a supervised, unsupervised or semi-supervised manner and solve them completely autonomously, provided a sufficient amount of data is available to train them. In DQN, the Q-tables known from classical Q-learning are substituted with neural networks in order to have them learn the values of <state, action>-pairs directly. Moreover, they are trained with an agent's experiences it has collected during its exploration of the environment. These experiences serve as a reusable form of training data. 

### DQN-Algorithm:
**The DQN-algorithm itself is composed of several components:**

#### Environment Navigation
The Q network is designed to map states to <state, action>-values. Thus, we can feed it our current state and then determine the best action as the one that has the largest estimated <state, action>-value. In practice, we then adopt an epsilon-greedy approach for selecting an action (epsilon-greedy means a random action is selected with probability epsilon in order to encourage early exploration, and the 'best' action is executed with probability (1-epsilon) in order to follow a policy that maximes reward deterministically.).

#### Q-network Learning
After we've collected enough experiences (i.e. <state, action, reward, state>-tuples) we start updating the model. This is achieved by sampling some of our experiences and then computing the empirically observed estimates of the <state, action>-values compared to those estimated from the model. The difference between these two is known as the TD-error. Subsequently, the model weights are updated in such a way as to reduce this error via the backpropagation-algorithm. The whole procedure is then repeated iteratively, effectively reducing the error over time until convergence is achieved.

A list of more refined versions of the DQN-algorithm can be found in the **"Possible Future Improvements"**-section together with brief descriptions.

## Empirical Results

**So far, we simply made use of the vanilla DQN.**
Below, you find some empirical results for our vanilla DQN.

### Reward-vs-Episodes-Plot

We found that after ~519 episodes, the agent can be considered to have 'solved' the environment as it attained an average reward over 100 episodes greater than 13.

A plot of score vs episodes is shown below:

![Rewards vs Episodes Plot](images/rewards_vs_episodes_plot.png)

### Gifs showing

#### the agent's performance before training

![Random Agent Performance](videos/random_agent.gif)

#### the agent's performance after training

![Trained Agent Performance](videos/test_agent.gif)

## Implementation

### Neural Network Architecture

As described above the architecture uses a vanilla DQN.
 
Overall, we chose a standard fully-connected, feed-forward neural network with the following parameters:

1.  Input layer consisting of 37 neurons, representing the 37-dimensional input state vector to our DQN.
2.  Two hidden layers with 32 neurons and ReLU-activation function each.
3.  One single output layer with 4 neurons representing the different actions (action_size = 4) and no (i.e. linear) activation function.

### Hyperparameters

Here, we would like to discuss the hyperparameters occurring in the DQN-algorithm briefly:

**n_episodes (int): maximum number of training episodes**\
We chose 2000 episodes.

**eps_start (float): starting value of epsilon for epsilon-greedy policy to select and action**\
High epsilon encourages exploration, while low epsilon encourages exploitation (policy). We chose epslon = 1.

**eps_end (float): minimum value of epsilon for the epsilon-greedy-policy**\
Epsilon should never reach 0 or else in the limit we might not explore all state-action pairs. Here, we chose a lower bound on epsilon of 0.01.

**eps_decay (float): multiplicative factor (per episode) for decreasing epsilon**\
we want epsilon to decay over the course of training as the agent transitions from exploration to exploitation. This was set to 0.995.

**GAMMA (float): discount rate**\
A value close to 1 will make the agent weight future rewards as strongly as immediate rewards, while a value close to 0 will make the agent only focus on immediate rewards. Here, we chose gamma to be 0.99.

**LR (float): model hyperparameter - learning rate**\
This determines the magnitude of model weight updates. If chosen too large a value, then the learning is likely to become unstable, while chosen too small a value, the model may never converge. Therefore, we chose LR to be 5e-4, which is a more or less common choice in Deep Learning.

**BATCH_SIZE (int): model hyperparameter - number of experiences sampled for a model minibatch**\
Too low will cause learning instability and poor convergence, too high can cause convergence to local optima. Here, we chose 64 as batch size.

**BUFFER_SIZE (int): replay buffer size**\
this is the size of the experience buffer and represents the size of the memory the agent can sample from for learning.

**TAU (float): how closely shall the target network follow the current network in terms of weights?**\
After every learning step, the target-network weights get updated with a fraction tau of the current network weights. Therefore, the target-network weights are a moving average over time of the current network weights. Here, we chose a value of 1e-3.

**UPDATE_EVERY (int): how often shall we update the network?**\
How many steps should pass before an update of the current network takes place. We chose every 4 timesteps.

## Possible Future Improvements

The following list of modifications and refined approaches may render themselves beneficial for obtaining even better performance than obtained with the DQN-algorithm presented and implemented here:**

### Double-DQN

Note, that when updating our Q-values we assume that the agent selects the action associated with the maximum <state, action>-value of the next timestep. However, since these <action, value>-estimates are likely to be noisy, taking the max is likely to overestimate their true value. One trick that comes to the rescue would be to use one model for selecting an action and another model to evaluate that action's value.

Read more: https://arxiv.org/abs/1509.06461

### Prioritised Experience replay

In order to produce training data we store all <state, action, reward, state>-tuples as experiences and then sample them randomly each time we update the model. Note, that some of these may be more valuable for learning than others. For instance, an agent may have plenty of experiences from the starting state but relatively little from more rare states. In this modification we use how 'surprising' an observed <state, action>-value is as a measure of how 'useful' learning from it is. Formally, this usefulness can be modeled by the absolute difference between the value we observed for it and the value our model assigns to it.

Read more: https://arxiv.org/abs/1511.05952

### Duel-DQN

In the context of traditional Q-learning, remember that only on single <state, action>-pair gets updated at every timestep. However, the action values might be very similar for many states and thus the former might be interesting for other states, too. Therefore, it would only be natural to split the "<state, action>-value learning task" into learning two separate tasks, i.e. learning the state value separately from learning the advantage of an action that was selected when the agent was in this state. This is the basic idea of the Duel-DQN approach.

Read more: https://arxiv.org/abs/1511.06581

### Rainbow

Additional modifications that might improve the algorithm further are the 3 modifications of the Rainbow implementation, which achieves state-of-the-art-performance in DQNs.

**These are:**

1.  Learning from multi-step bootstrap targets -  https://arxiv.org/abs/1602.01783
2.  Distributional DQN - https://arxiv.org/abs/1707.06887
3.  Noisy DQN - https://arxiv.org/abs/1706.10295
