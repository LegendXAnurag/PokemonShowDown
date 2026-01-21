"""
Training script for Pokemon RL using TorchRL and MAPPO/IPPO algorithms.
"""

import torch
import numpy as np
from pokemon_env import PokemonBattleEnv
from visualizer import PokemonVisualizer
import random
import time


class RandomPolicyTrainer:
    """
    Random policy trainer for Pokemon battle environment.
    This serves as a baseline and placeholder for full PPO/MAPPO/IPPO implementation with TorchRL.
    """
    
    def __init__(self, env, level=1, entropy_coef=0.01):
        """
        Initialize the trainer.
        
        Args:
            env: Pokemon battle environment
            level: Training level
            entropy_coef: Entropy coefficient for exploration
        """
        self.env = env
        self.level = level
        self.entropy_coef = entropy_coef
        
    def train(self, num_episodes=1000, visualize_every=100, save_every=500):
        """
        Train the agents.
        
        Args:
            num_episodes: Number of episodes to train
            visualize_every: Visualize every N episodes
            save_every: Save model every N episodes
        """
        print(f"Starting training for Level {self.level}")
        print(f"Entropy coefficient: {self.entropy_coef}")
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            observations, infos = self.env.reset()
            episode_reward = {agent: 0 for agent in self.env.agents}
            done = False
            step = 0
            
            while not done and step < self.env.max_steps:
                # Random policy (placeholder for actual PPO policy)
                actions = {agent: random.randint(0, 7) for agent in self.env.agents}
                
                observations, rewards, terminations, truncations, infos = self.env.step(actions)
                
                # Accumulate rewards
                for agent in rewards:
                    if agent in episode_reward:
                        episode_reward[agent] += rewards[agent]
                
                # Check if done
                done = all(terminations.values()) or all(truncations.values())
                step += 1
                
            # Log episode results
            avg_reward = np.mean(list(episode_reward.values()))
            episode_rewards.append(avg_reward)
            
            if episode % 10 == 0:
                recent_avg = np.mean(episode_rewards[-10:])
                print(f"Episode {episode}: Avg Reward: {recent_avg:.2f}")
                
            # Visualize
            if episode % visualize_every == 0 and episode > 0:
                self._visualize_episode(episode)
                
            # Save model
            if episode % save_every == 0 and episode > 0:
                self._save_model(episode)
                
        print(f"Training completed for Level {self.level}")
        
    def _visualize_episode(self, episode):
        """Visualize a single episode."""
        print(f"\n=== Visualizing Episode {episode} ===")
        
        visualizer = PokemonVisualizer(grid_size=self.env.grid_size, cell_size=60)
        observations, infos = self.env.reset()
        
        step = 0
        done = False
        
        while not done and step < 200:
            # Random policy
            actions = {agent: random.randint(0, 7) for agent in self.env.agents}
            observations, rewards, terminations, truncations, infos = self.env.step(actions)
            
            # Render
            visualizer.render(self.env, step)
            
            # Small delay
            time.sleep(0.1)
            
            # Handle events
            if not visualizer.handle_events():
                break
                
            done = all(terminations.values()) or all(truncations.values())
            step += 1
            
        # Save screenshot
        visualizer.save_screenshot(f"visualization_level{self.level}_ep{episode}.png")
        visualizer.close()
        
    def _save_model(self, episode):
        """Save model checkpoint."""
        print(f"Saving model checkpoint at episode {episode}")
        # Placeholder for actual model saving
        
        
def train_all_levels(num_episodes_per_level=1000):
    """
    Train agents on all 6 levels sequentially.
    
    Args:
        num_episodes_per_level: Number of episodes to train per level
    """
    for level in range(1, 7):
        print(f"\n{'='*50}")
        print(f"Training Level {level}")
        print(f"{'='*50}\n")
        
        # Create environment
        env = PokemonBattleEnv(level=level)
        
        # Create trainer with appropriate entropy coefficient
        # Higher entropy for exploration of action space
        # Note: Currently using random policy baseline
        entropy_coef = 0.05 if level <= 3 else 0.03
        trainer = RandomPolicyTrainer(env, level=level, entropy_coef=entropy_coef)
        
        # Train
        trainer.train(num_episodes=num_episodes_per_level, 
                     visualize_every=100,
                     save_every=500)
        
        print(f"\nLevel {level} training completed!\n")
        

if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Train on all levels
    train_all_levels(num_episodes_per_level=100)  # Reduced for testing
    
    print("\n" + "="*50)
    print("All levels training completed!")
    print("="*50)
