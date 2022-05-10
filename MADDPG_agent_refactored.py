import torch
import torch.nn.functional as F
import torch.nn as nn


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

# actor model definition
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(input_dim, Constants.NN_DIM)
        self.layer2 = nn.Linear(Constants.NN_DIM, Constants.NN_DIM)
        self.layer3 = nn.Linear(Constants.NN_DIM, output_dim)
        self.batch_norm = nn.BatchNorm1d(Constants.NN_DIM)

        # normalize weights
        self.layer1.weight.data.uniform_(-Constants.WEIGHT_NORM1, Constants.WEIGHT_NORM1)
        self.layer2.weight.data.uniform_(-Constants.WEIGHT_NORM1, Constants.WEIGHT_NORM1)
        self.layer3.weight.data.uniform_(-Constants.WEIGHT_NORM2, Constants.WEIGHT_NORM2)
    
    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = self.batch_norm(x)
        x = F.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        return x

# critic model definition
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(input_dim, Constants.NN_DIM)
        self.layer2 = nn.Linear(Constants.NN_DIM, Constants.NN_DIM)
        self.layer3 = nn.Linear(Constants.NN_DIM, 1)

        # normalize weights
        self.layer1.weight.data.uniform_(-Constants.WEIGHT_NORM1, Constants.WEIGHT_NORM1)
        self.layer2.weight.data.uniform_(-Constants.WEIGHT_NORM1, Constants.WEIGHT_NORM1)
        self.layer3.weight.data.uniform_(-Constants.WEIGHT_NORM2, Constants.WEIGHT_NORM2)
    
    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class DDPG():
    def __init__(self, idx):
        self.idx = idx
        # the actor models
        self.actor_local = Actor(Constants.ACTION_DIM, Constants.NN_DIM).to(Constants.DEVICE)
        self.actor_target = Actor(Constants.ACTION_DIM, Constants.NN_DIM).to(Constants.DEVICE)
        
        # the critic models
        self.critic_local = Critic(Constants.NN_DIM).to(Constants.DEVICE)
        self.critic_target = Critic(Constants.NN_DIM).to(Constants.DEVICE)
        
        # set the optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=Constants.ALPHA_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=Constants.ALPHA_CRITIC)

        # copy the weights from the local to the target
        # critic
        for target, local in zip(self.critic_target.parameters(), self.critic_local.parameters()):
            target.data.copy_(local.data)
        
        # actor
        for target, local in zip(self.actor_target.parameters(), self.actor_local.parameters()):
            target.data.copy_(local.data)
    
    def action(self, state):
        state = torch.from_numpy(state).float().to(Constants.DEVICE)    
        self.actor_local.eval()
        with torch.no_grad(): 
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        
