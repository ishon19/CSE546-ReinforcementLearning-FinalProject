import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


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
    NOISE1 = 0.1
    NOISE2 = 0.1
    NOISE3 = 30000
    STATE_DIM = 24
    ACTION_DIM = 2
    NUM_AGENTS = 2
    BATCH_SIZE = 256

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
        self.ac_actor_local = Actor(Constants.STATE_DIM, Constants.ACTION_DIM).to(Constants.DEVICE)
        self.ac_actor_target = Actor(Constants.STATE_DIM, Constants.ACTION_DIM).to(Constants.DEVICE)
        input_size = (Constants.STATE_DIM + Constants.ACTION_DIM) * Constants.NUM_AGENTS

        # critic pair
        self.ac_critic_local = Critic(input_size).to(Constants.DEVICE)
        self.ac_critic_target = Critic(input_size).to(Constants.DEVICE)

class DDPG():
    def __init__(self, idx):
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

        # copy the weights from the local to the target
        # critic
        for target, local in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target.data.copy_(local.data)

        # actor
        for target, local in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target.data.copy_(local.data)

    def _transform_state(self, state):
        state_trans = torch.from_numpy(state).float().to(Constants.DEVICE)
        self.actor_local.eval()
        return state_trans

    def action(self, state):
        state = self._transform_state(state)
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        return np.clip(action, -1, 1)

    def _get_idx_actions(self, idx, actions_next):
        agent_idx = torch.tensor([idx]).to(Constants.DEVICE)
        next_actions = torch.cat(actions_next, dim=1).to(Constants.DEVICE)
        return agent_idx, next_actions

    def _get_q_values(self, idx, s, a,  r, d, target_q_next):
        exp_q, tar_q = self.critic_local(s, a), r.index_select(
            1, idx) + Constants.GAMMA * target_q_next * (1 - d.index_select(1, idx))
        return exp_q, tar_q

    def _get_expected_actions_loss(self, s,  a, actions):
        pred = [a.detach() if i != self.idx else a for i, a in actions]
        pred = torch.cat(pred, dim=1)
        return -self.critic_local(s, pred).mean(), pred

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
        exp_actions, loss_actor = self._get_expected_actions_loss(
            s, a, actions)
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
                #self.idx = idx
                self.steps = 0  
                
                              
