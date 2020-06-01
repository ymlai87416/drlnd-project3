[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"

[image3]: ./images/critic.PNG "critic"
[image4]: ./images/actor.PNG "actor"
[image5]: ./images/result.png "result"

# Introduction
For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of `+0.1`.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of `-0.01`.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of `8` variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of `+0.5` (over `100` consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


# Environment

Agent is trained in the following environment

* Python 3
* Pytorch 0.4.1 cuda version 9.2
* Ubuntu 18.04
* AMD Ryzen Threadripper 2950X 16-Core Processor
* Nvidia RTX 2800Ti

# Agent

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

```
Number of agents: 2
Size of each action: 2
There are 2 agents. Each observes a state with length: 24
The state for the first agent looks like: 
[[ 0.          0.          0.          0.          0.          0.
   0.          0.          0.          0.          0.          0.
   0.          0.          0.          0.         -6.4669857  -1.5
   0.          0.         -6.83172083  6.          0.          0.        ]]
```

# Algorithm

MADDPG [1] is used to train the agent.

## Actor-Critic

In file `model.py`, the class `Actor` implements the actor network, and the class `Critic` implements the critic network.

Each DDPG agent create 4 networks, they are actor_local, actor_target, critic_local and critic_target (refer to class `Agent` of file `ddpg_agent.py`). Learning is done on actor_local and critic_local, and soft update is used to gradually apply the weighting of the local network to the target network.

### Actor network

The network acts as a policy base method, the input is current state, the output is an optimal action.

```
Actor(
  (fc1): Linear(in_features=24, out_features=256, bias=True)
  (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=2, bias=True)
)
```

### Critic network

The critic network as a value base network, input is the current state (as input of fcs1 layer) and the actions taken by 2 agents (as input of fc2 layer), output is the expected reward.

```
Critic(
  (fcs1): Linear(in_features=24, out_features=256, bias=True)
  (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=260, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=1, bias=True)
)
```

## Centralized critic for each agent

The critic network explicitly uses the decision-making policies of other agents, agents can learn approximate models of other agents online and effectively use them in their own policy learning procedure.

Learning function is implemented in file maddpg_agent.py. In class `MADDPGAgent` function `learn`. 

The section of code updating the critic network is Line 81-98. The next action of each agent and the real reward is used to improve the critic network to provide a better Q-function.

The implementation refers to the equation (6) of the paper [1].

![Critic network update][image3]


The section of code updating the actor network is Line 100-111. The next action of each agent and Q-function from critic network is considered to find a better policy function.

The implementation refers to the equation (5) of the paper [1].

![Actor network update][image4]

Soft update is then applied to both actor and critic network.

## Reply Buffer

In file `maddpg_agent.py`, the class `ReplayBuffer` implements the replay buffer, the replay buffer save all experience experienced by 2 agents. Random experiences are retrieved from time to time to train both actor and critic network.

## Ornstein-Uhlenbeck Process for Noise generation

In file `maddpg_agent.py`, the class `OUNoise` implements the noise generation process to allow DDPG to explore around the optimal action given by the actor network. Please see the hyperparameter section for parameter value selection.

## Fine tuning

In this implementation, several techniques are employed.

Batch normalization is used to allow learn effectively across different types of units.

Gradient clipping is also used to keep the score stable across 100 episodes by avoiding the network making updates that are too far away from the current weighting. Although gradient clipping is applied, the agent starts to perform horribly starting around episode 2000 and show no evidence of recovery.

The model is trained while the Ornstein-Uhlenbeck process is turned of. It helps these agents to archive score > 0.5 given the current parameter, if I turn on the Ornstein-Uhlenbeck process, agents are more likely to hit the ball out of the table, and the score seldom greater than 0.3.

# Hyperparameters

Here is the network structure of actor

```
Actor(
  (fc1): Linear(in_features=24, out_features=256, bias=True)
  (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=2, bias=True)
)
```

and critic

```
Critic(
  (fcs1): Linear(in_features=24, out_features=256, bias=True)
  (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (fc2): Linear(in_features=260, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=1, bias=True)
)
```

Max timestep for training = 300.

Discount factor is of 𝛾 = 0.95

Training occurs at every 2 timesteps and updates the network for 10 times. Mini batch size = 128 experiences.

Actor and critic network initialization:

Final layer: uniform between [-3e-3, 3e-3]

Other layer: uniform between [-1/sqrt(f), 1/sqrt(f)] where f is the fan-in of the layer.

the learning rate for actor is 10-4 and critic is 10-3.

Soft update 𝜏 = 0.001

No weight decay on neither actor nor critic network

Ornstein-Uhlenbeck process: μ = 0, 𝜃 = 0.15, 𝜎 = 0.08

Noise decay at a rate 0.995 per each learning

# Result

The graph shows that the agents get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).

![Result][image5]

The average score over 100 episodes across the time is
```
Episode 100	Average Score: 0.00
Episode 200	Average Score: 0.00
Episode 300	Average Score: 0.01
Episode 400	Average Score: 0.03
Episode 500	Average Score: 0.08
Episode 600	Average Score: 0.13
Episode 700	Average Score: 0.43
Episode 800	Average Score: 0.58
Episode 900	Average Score: 0.62
Episode 1000	Average Score: 0.45
Episode 1100	Average Score: 0.52
Episode 1200	Average Score: 0.39
Episode 1300	Average Score: 0.49
Episode 1400	Average Score: 0.65
Episode 1500	Average Score: 0.59
Episode 1600	Average Score: 0.55
Episode 1700	Average Score: 0.38
Episode 1800	Average Score: 0.41
Episode 1900	Average Score: 0.34
Episode 2000	Average Score: 0.47
```

At episode 800-900, 110 and 1400-1600, the agents get an average score bigger than +0.5


Here is trained model in action [link](https://youtu.be/Y33TTszuRwI)

```
Score (max over agents) from episode 1: 2.600000038743019
Score (max over agents) from episode 2: 0.5000000074505806
Score (max over agents) from episode 3: 1.600000023841858
Score (max over agents) from episode 4: 2.600000038743019
Score (max over agents) from episode 5: 0.4000000059604645
Score (max over agents) from episode 6: 0.10000000149011612
Score (max over agents) from episode 7: 0.30000000447034836
Score (max over agents) from episode 8: 0.10000000149011612
Score (max over agents) from episode 9: 2.600000038743019
```


# Future works

Here is some idea to further improve the performance and the learning efficiency of the agent.

* Implement prioritized experience reply buffer.
* Turn on Ornstein-Uhlenbeck process during training. It may make agent more resillient to other agent mistake[2]. You can see that either agents work perfectly and complete 300 timestep together or they fail very early.

# Reference

[1] Lowe, Ryan, et al. "Multi-agent actor-critic for mixed cooperative-competitive environments." Advances in neural information processing systems. 2017.

[2] Gleave, Adam, et al. "Adversarial policies: Attacking deep reinforcement learning." arXiv preprint arXiv:1905.10615 (2019).