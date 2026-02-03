# model.py
import torch
import torch.nn as nn
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PokemonAgent(nn.Module):
    def __init__(self, obs_dim, action_dim=6):
        super(PokemonAgent, self).__init__()
        
        # [CHANGE] Network scales with obs_dim
        # Old obs_dim ~ 50. New obs_dim ~ 2 + (16*5) = 82
        
        self.network = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 512)),
            nn.LayerNorm(512),
            nn.LeakyReLU(negative_slope=0.01),
            
            layer_init(nn.Linear(512, 256)),
            nn.LayerNorm(256),
            nn.LeakyReLU(negative_slope=0.01),
            
            layer_init(nn.Linear(256, 256)),
            nn.LayerNorm(256),
            nn.LeakyReLU(negative_slope=0.01),
            
            layer_init(nn.Linear(256, 128)),
            nn.LayerNorm(128),
            nn.LeakyReLU(negative_slope=0.01),
        )

        self.actor = layer_init(nn.Linear(128, action_dim), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)

    def get_value(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        hidden = self.network(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None, action_mask=None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        hidden = self.network(x)
        logits = self.actor(hidden)

        if action_mask is not None:
            if not isinstance(action_mask, torch.Tensor):
                action_mask = torch.tensor(action_mask, device=x.device)
            min_value = torch.tensor(-1e9, device=x.device)
            logits = torch.where(action_mask > 0.5, logits, min_value)

        probs = torch.distributions.Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

if __name__ == "__main__":
    # Sanity Check for new Lidar size
    # 2 self + 16 rays * 5 channels = 82
    model = PokemonAgent(obs_dim=82)
    print(model)
    dummy = torch.randn(1, 82)
    a, lp, e, v = model.get_action_and_value(dummy)
    print(f"Sanity Check: Action {a.item()}, Value {v.item()}")