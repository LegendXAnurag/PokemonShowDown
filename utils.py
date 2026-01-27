"""
Utility functions for training and evaluation
"""
import torch
import numpy as np
import os
import json
from datetime import datetime

class ExperienceBuffer:
    """
    Buffer for storing experiences during rollout
    """
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        
    def store(self, obs, action, reward, done, value, log_prob):
        """Store a single transition"""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.log_probs.append(log_prob)
    
    def get(self):
        """Get all stored experiences"""
        return {
            'observations': torch.FloatTensor(np.array(self.observations)),
            'actions': torch.LongTensor(np.array(self.actions)),
            'rewards': torch.FloatTensor(np.array(self.rewards)),
            'dones': torch.FloatTensor(np.array(self.dones)),
            'values': torch.FloatTensor(np.array(self.values)),
            'log_probs': torch.FloatTensor(np.array(self.log_probs))
        }
    
    def clear(self):
        """Clear buffer"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
    
    def __len__(self):
        return len(self.observations)


def compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
    """
    Compute Generalized Advantage Estimation
    
    Args:
        rewards: Tensor of rewards
        values: Tensor of value estimates
        dones: Tensor of done flags
        next_value: Value estimate for next state
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        advantages: GAE advantages
        returns: Discounted returns
    """
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    # Add next_value to values
    values = torch.cat([values, next_value.unsqueeze(0)])
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value_t = values[t + 1]
        else:
            next_non_terminal = 1.0 - dones[t]
            next_value_t = values[t + 1]
        
        delta = rewards[t] + gamma * next_value_t * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
    
    returns = advantages + values[:-1]
    
    return advantages, returns


def normalize_advantages(advantages):
    """Normalize advantages to have mean 0 and std 1"""
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


class Logger:
    """Simple logger for training metrics"""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, 'training_log.json')
        self.metrics = []
        
    def log(self, step, metrics):
        """Log metrics for a step"""
        entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics.append(entry)
        
        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def print_metrics(self, step, metrics):
        """Print metrics to console"""
        print(f"\n{'='*60}")
        print(f"Step: {step}")
        print(f"{'='*60}")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:30s}: {value:10.4f}")
            else:
                print(f"{key:30s}: {value}")
        print(f"{'='*60}\n")


def save_checkpoint(model, optimizer, step, level, save_dir):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'level': level
    }
    
    filepath = os.path.join(save_dir, f'checkpoint_level_{level}_step_{step}.pt')
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['step'], checkpoint['level']


def get_linear_schedule(start_value, end_value, total_steps):
    """Create linear learning rate schedule"""
    def schedule(step):
        fraction = min(step / total_steps, 1.0)
        return start_value + (end_value - start_value) * fraction
    return schedule


class RewardTracker:
    """Track episode rewards for each agent"""
    def __init__(self):
        self.episode_rewards = {}
        self.completed_episodes = []
        
    def update(self, agent, reward):
        """Update reward for agent"""
        if agent not in self.episode_rewards:
            self.episode_rewards[agent] = 0
        self.episode_rewards[agent] += reward
    
    def finish_episode(self):
        """Mark episode as complete and reset"""
        if self.episode_rewards:
            avg_reward = np.mean(list(self.episode_rewards.values()))
            self.completed_episodes.append(avg_reward)
        self.episode_rewards = {}
    
    def get_stats(self):
        """Get reward statistics"""
        if not self.completed_episodes:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
        
        rewards = np.array(self.completed_episodes)
        return {
            'mean': rewards.mean(),
            'std': rewards.std(),
            'min': rewards.min(),
            'max': rewards.max()
        }
    
    def reset(self):
        """Reset all tracking"""
        self.episode_rewards = {}
        self.completed_episodes = []


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)