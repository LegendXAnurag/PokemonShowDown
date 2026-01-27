"""
Neural network architectures for MAPPO/IPPO
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NETWORK_CONFIG, NUM_ACTIONS

class ActorCritic(nn.Module):
    def __init__(self, obs_dim):
        """
        Actor-Critic network for PPO
        
        Args:
            obs_dim: Dimension of observation space
        """
        super(ActorCritic, self).__init__()
        
        hidden_dim = NETWORK_CONFIG['hidden_dim']
        num_layers = NETWORK_CONFIG['num_layers']
        
        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            in_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, NUM_ACTIONS)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)
    
    def forward(self, obs):
        """
        Forward pass
        
        Args:
            obs: Observation tensor
            
        Returns:
            action_logits: Logits for action distribution
            value: State value estimate
        """
        features = self.feature_extractor(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        
        return action_logits, value
    
    def get_action_and_value(self, obs, action=None):
        """
        Get action, log probability, entropy, and value
        
        Args:
            obs: Observation tensor
            action: Optional action (if provided, compute log prob for this action)
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            entropy: Entropy of action distribution
            value: State value estimate
        """
        action_logits, value = self.forward(obs)
        
        # Create categorical distribution
        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value
    
    def get_value(self, obs):
        """
        Get value estimate only
        
        Args:
            obs: Observation tensor
            
        Returns:
            value: State value estimate
        """
        features = self.feature_extractor(obs)
        value = self.critic(features)
        return value


class CentralizedCritic(nn.Module):
    """
    Centralized critic for MAPPO (optional)
    Takes global state as input
    """
    def __init__(self, global_obs_dim):
        """
        Initialize centralized critic
        
        Args:
            global_obs_dim: Dimension of global observation
        """
        super(CentralizedCritic, self).__init__()
        
        hidden_dim = NETWORK_CONFIG['hidden_dim']
        
        self.network = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0)
    
    def forward(self, global_obs):
        """
        Forward pass
        
        Args:
            global_obs: Global observation tensor
            
        Returns:
            value: State value estimate
        """
        return self.network(global_obs)


# Import numpy for initialization
import numpy as np