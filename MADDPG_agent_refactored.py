from collections import deque, namedtuple
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import copy


class Constants:
    """Constants for the game."""
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NN_DIM = 256
    WEIGHT_NORM1 = 1e-2
    WEIGHT_NORM2 = 3e-3
    ALPHA_ACTOR = 1e-4
    ALPHA_CRITIC = 1e-3
    WEIGHT_DECAY = 0.0
    ACTION_DIM = 2
    SEED = 777
    GAMMA = 0.99
    TAU = 1e-3
    ACTION_DIM = 2
    BUFFER_SIZE = 10000
    UPDATE_INTERVAL = 2
    NOISE1 = 1.0
    NOISE2 = 0.1
    NOISE3 = 30000
    STATE_DIM = 24
    ACTION_DIM = 2
    NUM_AGENTS = 2
    BATCH_SIZE = 256
    MU = 0.0

# actor model network


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(input_dim, Constants.NN_DIM)
        self.layer2 = nn.Linear(Constants.NN_DIM, Constants.NN_DIM)
        self.layer3 = nn.Linear(Constants.NN_DIM, output_dim)
        self.batch_norm = nn.BatchNorm1d(Constants.NN_DIM)

        # normalize weights
        self.layer1.weight.data.uniform_(-Constants.WEIGHT_NORM1,
                                         Constants.WEIGHT_NORM1)
        self.layer2.weight.data.uniform_(-Constants.WEIGHT_NORM1,
                                         Constants.WEIGHT_NORM1)
        self.layer3.weight.data.uniform_(-Constants.WEIGHT_NORM2,
                                         Constants.WEIGHT_NORM2)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = self.batch_norm(x)
        x = F.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        return x


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(input_dim, Constants.NN_DIM)
        self.layer2 = nn.Linear(Constants.NN_DIM, Constants.NN_DIM)
        self.layer3 = nn.Linear(Constants.NN_DIM, 1)

        # normalize weights
        self.layer1.weight.data.uniform_(-Constants.WEIGHT_NORM1,
                                         Constants.WEIGHT_NORM1)
        self.layer2.weight.data.uniform_(-Constants.WEIGHT_NORM1,
                                         Constants.WEIGHT_NORM1)
        self.layer3.weight.data.uniform_(-Constants.WEIGHT_NORM2,
                                         Constants.WEIGHT_NORM2)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# critic model definition
class ActorCritic(nn.Module):
    def __init__(self):
        # actor pair
        self.ac_actor_local = Actor(
            Constants.STATE_DIM, Constants.ACTION_DIM).to(Constants.DEVICE)
        self.ac_actor_target = Actor(
            Constants.STATE_DIM, Constants.ACTION_DIM).to(Constants.DEVICE)
        input_size = (Constants.STATE_DIM +
                      Constants.ACTION_DIM) * Constants.NUM_AGENTS

        # critic pair
        self.ac_critic_local = Critic(input_size).to(Constants.DEVICE)
        self.ac_critic_target = Critic(input_size).to(Constants.DEVICE)


class DDPG():
    def __init__(self, idx):
        self.mu = np.ones(Constants.ACTION_DIM) * Constants.MU
        self.idx = idx
        # the actor models
        self.actor_local = Actor(Constants.ACTION_DIM,
                                 Constants.NN_DIM).to(Constants.DEVICE)
        self.actor_target = Actor(
            Constants.ACTION_DIM, Constants.NN_DIM).to(Constants.DEVICE)

        # the critic models
        self.critic_local = Critic(Constants.NN_DIM).to(Constants.DEVICE)
        self.critic_target = Critic(Constants.NN_DIM).to(Constants.DEVICE)

        # set the optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor_local.parameters(), lr=Constants.ALPHA_ACTOR)
        self.critic_optimizer = torch.optim.Adam(
            self.critic_local.parameters(), lr=Constants.ALPHA_CRITIC)

        # noise generator
        self.noise_state = copy.copy(Constants.MU)

        # copy the weights from the local to the target
        # critic
        for target, local in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target.data.copy_(local.data)

        # actor
        for target, local in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target.data.copy_(local.data)

    def sample_noise(self):
        x, dx = self.noise_state, Constants.NOISE_THETA * \
            (self.mu - x) + Constants.NOISE_SIGMA * np.random.randn(self.size)
        self.noise_state = x + dx
        return self.noise_state

    def _transform_state(self, state):
        state_trans = torch.from_numpy(state).float().to(Constants.DEVICE)
        self.actor_local.eval()
        return state_trans

    def action(self, state):
        state = self._transform_state(state)
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        self.val_noise = self.sample_noise() * Constants.NOISE1
        return np.clip(action, -1, 1)

    def _get_idx_actions(self, idx, actions_next):
        agent_idx = torch.tensor([idx]).to(Constants.DEVICE)
        next_actions = torch.cat(actions_next, dim=1).to(Constants.DEVICE)
        return agent_idx, next_actions

    def _get_q_values(self, idx, s, a,  r, d, target_q_next):
        exp_q, tar_q = self.critic_local(s, a), r.index_select(
            1, idx) + Constants.GAMMA * target_q_next * (1 - d.index_select(1, idx))
        return exp_q, tar_q

    def _get_expected_loss(self, s,  a, actions):
        pred = [a.detach() if i != self.idx else a for i, a in actions]
        pred = torch.cat(pred, dim=1)
        return -self.critic_local(s, pred).mean()

    def update(self, idx, experiences, actions_next, actions):
        # get the states, actions, rewards, and don'ts from the experiences
        s, a, r, n, d = experiences
        self.critic.optimizer.zero_grad()
        agent_idx, next_actions = self._get_idx_actions(idx, actions_next)

        # get the Q values from the critic
        with torch.no_grad():
            target_q_next = self.critic_target(next_actions)
        q_expected, q_targets = self._get_q_values(
            idx, s, a, r, d, target_q_next)

        # compute the critic loss
        loss_critic = F.mse_loss(q_targets.detach(), q_expected)
        loss_critic.backward()  # compute the gradients
        self.critic_optimizer.step()  # update the critic

        # compute the actor loss
        self.actor_optimizer.zero_grad()
        loss_actor = self._get_expected_loss(s, a, actions)
        loss_actor.backward()
        self.actor_optimizer.step()

        # update the target networks
        for target, local in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target.data.copy_(Constants.TAU * local.data +
                              (1 - Constants.TAU) * target.data)
        for target, local in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target.data.copy_(Constants.TAU * local.data +
                              (1 - Constants.TAU) * target.data)

class MADDPGAgent():
    def __init__(self):
        np.random.seed(Constants.SEED)
        random.seed(Constants.SEED)
        self.steps = 0
        model_list = [ActorCritic()] * Constants.NUM_AGENTS
        agent_list = [DDPG(i) for i in range(Constants.NUM_AGENTS)]
        self.experience = namedtuple("ReplayBuffer", field_names=[
                                        "s", "a", "r", "n", "d"])
        self.memory = deque(maxlen=Constants.BUFFER_SIZE)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(self.experience(
            state, action, reward, next_state, done))

    def _to_tensor(self, obj):
        return torch.from_numpy(obj).float().to(Constants.DEVICE).float().to(Constants.DEVICE)

    def sampleMemory(self):
        memory_sample = random.sample(
            self.memory, Constants.BATCH_SIZE)
        s, a = self._to_tensor(
            np.vstack([e.s for e in memory_sample if e is not None])),
        self._to_tensor(
            np.vstack([e.a for e in memory_sample if e is not None]))
        r, n = self._to_tensor(
            np.vstack([e.r for e in memory_sample if e is not None])),
        self._to_tensor(
            np.vstack([e.n for e in memory_sample if e is not None]))
        d = self._to_tensor(
            np.vstack([e.d for e in memory_sample if e is not None]))
        return (s, a, r, n, d)
    
    def step(self, params):
        state, action, reward, next_state, done = params
        state = state.reshape((1, -1))
        next_state = next_state.reshape((1, -1))
        self.remember(state, action, reward, next_state, done)
        self.steps += 1
        if self.steps % Constants.UPDATE_INTERVAL == 0:
            if len(self.memory) > Constants.BATCH_SIZE:
                exps = [self.sampleMemory() for _ in range(Constants.NUM_AGENTS)]
                self.update(exps)
    
    def action(self, states):
        act_list = []
        for a,s  in zip(self.agent_list, states):
            action = a.action(s)
            act_list.append(action)
        return np.array(act_list).reshape((-1, 1))
    
    def update(self, exps):
        next_act_list, act_list = [], []
        for i, a in enumerate(self.agent_list):
            s, a, r, n, d = exps[i]
            a_idx = torch.tensor([i]).to(Constants.DEVICE)
            state = s.reshape(1, 2, 24).index_select(1, a_idx).squeeze(1)
            action = a.actor_local(state)
            act_list.append(action) 
            next_state = n.reshape(1, 2, 24).index_select(1, a_idx).squeeze(1)
            next_action = a.actor_target(next_state)
            next_act_list.append(next_action)
        
        for i, a in enumerate(self.agent_list):
            a.update((s, a, r, n, d))


