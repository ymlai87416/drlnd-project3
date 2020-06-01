import numpy as np
from ddpg_agent import Agent 
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
from collections import namedtuple, deque
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.95            # discount factor
TAU = 1e-3              # for soft update of target parameters

LEARN_EVERY = 2        # learn every X timestep
LEARN_NUM = 10         #   for Y time
NOISE_REDUCTION = 0.9995

class MADDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, num_agent, state_size, action_size, random_seed):
        self.maddpg_agent = []
        self.num_agent = num_agent

        for i  in range(num_agent):
            self.maddpg_agent.append(Agent(num_agent, state_size, action_size, random_seed))

        # Replay memory
        self.memory = ReplayBuffer(num_agent, action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    

    def step(self, state, action, reward, next_state, done, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)      

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and ((timestep % LEARN_EVERY) == 0):
            for _ in range(LEARN_NUM):
                for i in range(self.num_agent):
                    experiences = self.memory.sample()
                    self.learn(i, experiences, GAMMA)

    def act(self, states, add_noise=True):
        result = []
        for i in range(self.num_agent):
            result.append(self.maddpg_agent[i].act(states[i:i+1], add_noise))

        result = np.concatenate(result)
        return result


    def reset(self):
        for i  in range(self.num_agent):
            self.maddpg_agent[i].reset()


    def learn(self, agent_number, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        state = states[agent_number]
        action = actions[agent_number]
        reward = rewards[agent_number]
        next_state = next_states[agent_number]
        done = dones[agent_number]

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = [self.maddpg_agent[i].actor_target(states[i]).detach()
                            for i in range(self.num_agent)]
        actions_next = torch.cat(actions_next, dim=1)
        Q_targets_next = self.maddpg_agent[agent_number].critic_target(next_state, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = reward + (gamma * Q_targets_next * (1 - done))
        # Compute critic loss
        actions_exp = torch.cat(actions, dim=1)
        Q_expected = self.maddpg_agent[agent_number].critic_local(state, actions_exp)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.maddpg_agent[agent_number].critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.maddpg_agent[agent_number].critic_local.parameters(), 1)
        self.maddpg_agent[agent_number].critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = [ self.maddpg_agent[i].actor_local(state) if i == agent_number \
                   else self.maddpg_agent[i].actor_local(state).detach()
                   for i, state in enumerate(states) ]
        actions_pred = torch.cat(actions_pred, dim=1)

        actor_loss = -self.maddpg_agent[agent_number].critic_local(state, actions_pred).mean()
        # Minimize the loss
        self.maddpg_agent[agent_number].actor_optimizer.zero_grad()
        actor_loss.backward()
        self.maddpg_agent[agent_number].actor_optimizer.step()

        # ---------------------------- reduce noise ---------------------------- #
        for i in range(self.num_agent):
            self.maddpg_agent[i].noise_decay = self.maddpg_agent[i].noise_decay * NOISE_REDUCTION

        # ----------------------- update target networks ----------------------- #
        self.maddpg_agent[agent_number].soft_update(self.maddpg_agent[agent_number].critic_local, self.maddpg_agent[agent_number].critic_target, TAU)
        self.maddpg_agent[agent_number].soft_update(self.maddpg_agent[agent_number].actor_local, self.maddpg_agent[agent_number].actor_target, TAU)                     

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, num_agent, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.num_agent = num_agent
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""
        e = self.experience(states, actions, rewards, next_states, dones)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in range(self.num_agent):
            states.append(torch.from_numpy(np.vstack([e.state[i] for e in experiences if e is not None])).float().to(device))
            actions.append(torch.from_numpy(np.vstack([e.action[i] for e in experiences if e is not None])).float().to(device))
            rewards.append(torch.from_numpy(np.vstack([e.reward[i] for e in experiences if e is not None])).float().to(device))
            next_states.append(torch.from_numpy(np.vstack([e.next_state[i] for e in experiences if e is not None])).float().to(device))
            dones.append(torch.from_numpy(np.vstack([e.done[i] for e in experiences if e is not None]).astype(np.uint8)).float().to(device))

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
