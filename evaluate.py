"""
Evaluation script for trained Pokemon agents
"""
import torch
import numpy as np
import argparse
import time
from collections import defaultdict

from environment import PokemonShowdownEnv
from networks import ActorCritic
from utils import load_checkpoint
from config import RENDER_CONFIG


class Evaluator:
    """Evaluate trained agents"""
    
    def __init__(self, env, models, device='cpu'):
        """
        Initialize evaluator
        
        Args:
            env: Environment
            models: Dictionary of agent -> model
            device: torch device
        """
        self.env = env
        self.models = models
        self.device = device
        
    def evaluate(self, num_episodes=10, render=True, deterministic=False, delay=0.05):
        """
        Evaluate agents for multiple episodes
        
        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render
            deterministic: Use deterministic actions (argmax) instead of sampling
            delay: Delay between steps for visualization (seconds)
            
        Returns:
            Dictionary of evaluation metrics
        """
        episode_rewards = defaultdict(list)
        episode_lengths = []
        win_rates = defaultdict(int)
        damage_stats = defaultdict(lambda: {'dealt': [], 'taken': []})
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            
            observations, _ = self.env.reset()
            episode_reward = defaultdict(float)
            done = False
            step = 0
            
            while not done:
                actions = {}
                
                for agent in self.env.agents:
                    obs_tensor = torch.FloatTensor(observations[agent]).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        action_logits, _ = self.models[agent].forward(obs_tensor)
                        
                        if deterministic:
                            action = torch.argmax(action_logits, dim=-1)
                        else:
                            probs = torch.softmax(action_logits, dim=-1)
                            action = torch.multinomial(probs, 1)
                        
                        actions[agent] = action.item()
                
                observations, rewards, terminations, truncations, _ = self.env.step(actions)
                
                # Render after step to show beam animations
                if render:
                    self.env.render()
                    if delay > 0:
                        time.sleep(delay)
                
                for agent in self.env.agents:
                    episode_reward[agent] += rewards.get(agent, 0)
                
                done = all(terminations.values()) or all(truncations.values())
                step += 1
            
            # Final render to show end state
            if render:
                self.env.render()
                time.sleep(0.5)  # Pause to show final state
            
            # Record episode stats
            episode_lengths.append(step)
            
            for agent, reward in episode_reward.items():
                episode_rewards[agent].append(reward)
            
            # Collect damage statistics
            for agent in self.env.possible_agents:
                pokemon = self.env.pokemons[agent]
                damage_stats[agent]['dealt'].append(pokemon.total_damage_dealt)
                damage_stats[agent]['taken'].append(pokemon.total_damage_taken)
            
            # Determine winner(s)
            alive_agents = [agent for agent in self.env.possible_agents 
                          if self.env.pokemons[agent].alive]
            
            if len(alive_agents) == 1:
                winner = alive_agents[0]
                win_rates[winner] += 1
                print(f"Winner: {winner}")
            elif len(alive_agents) > 1:
                # Team victory or draw
                print(f"Multiple survivors: {alive_agents}")
                for agent in alive_agents:
                    win_rates[agent] += 1 / len(alive_agents)
            else:
                print("No survivors (draw)")
            
            # Print episode summary
            print(f"Episode length: {step} steps")
            for agent in self.env.possible_agents:
                pokemon = self.env.pokemons[agent]
                status = "ALIVE" if pokemon.alive else "DEAD"
                reward = episode_reward.get(agent, 0)
                print(f"{agent} ({pokemon.name}): {reward:.2f} | HP: {pokemon.health:.0f}/{pokemon.max_health} | "
                      f"Damage: {pokemon.total_damage_dealt:.0f} dealt, {pokemon.total_damage_taken:.0f} taken | {status}")
        
        # Compute statistics
        stats = {
            'episode_lengths': {
                'mean': np.mean(episode_lengths),
                'std': np.std(episode_lengths),
                'min': np.min(episode_lengths),
                'max': np.max(episode_lengths)
            },
            'episode_rewards': {},
            'win_rates': {},
            'damage_stats': {}
        }
        
        for agent in self.env.possible_agents:
            if agent in episode_rewards:
                rewards = episode_rewards[agent]
                stats['episode_rewards'][agent] = {
                    'mean': np.mean(rewards),
                    'std': np.std(rewards),
                    'min': np.min(rewards),
                    'max': np.max(rewards)
                }
            
            stats['win_rates'][agent] = win_rates[agent] / num_episodes
            
            # Damage statistics
            stats['damage_stats'][agent] = {
                'dealt': {
                    'mean': np.mean(damage_stats[agent]['dealt']),
                    'total': np.sum(damage_stats[agent]['dealt'])
                },
                'taken': {
                    'mean': np.mean(damage_stats[agent]['taken']),
                    'total': np.sum(damage_stats[agent]['taken'])
                }
            }
        
        return stats


def print_evaluation_stats(stats):
    """Pretty print evaluation statistics"""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print("\nEpisode Lengths:")
    for key, value in stats['episode_lengths'].items():
        print(f"  {key:10s}: {value:8.2f}")
    
    print("\nEpisode Rewards:")
    for agent, rewards in stats['episode_rewards'].items():
        print(f"\n  {agent}:")
        for key, value in rewards.items():
            print(f"    {key:10s}: {value:8.2f}")
    
    print("\nWin Rates:")
    for agent, rate in stats['win_rates'].items():
        print(f"  {agent:15s}: {rate:6.2%}")
    
    print("\nDamage Statistics:")
    for agent, damage in stats['damage_stats'].items():
        print(f"\n  {agent}:")
        print(f"    Damage Dealt (avg): {damage['dealt']['mean']:8.2f}")
        print(f"    Damage Dealt (tot): {damage['dealt']['total']:8.2f}")
        print(f"    Damage Taken (avg): {damage['taken']['mean']:8.2f}")
        print(f"    Damage Taken (tot): {damage['taken']['total']:8.2f}")
        if damage['taken']['mean'] > 0:
            ratio = damage['dealt']['mean'] / damage['taken']['mean']
            print(f"    Dealt/Taken Ratio:  {ratio:8.2f}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=str, default='level_1',
                       choices=[f'level_{i}' for i in range(1, 7)],
                       help='Training level')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory containing checkpoints')
    parser.add_argument('--num_episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use: auto, cuda, or cpu')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic actions')
    parser.add_argument('--no_render', action='store_true',
                       help='Disable rendering')
    parser.add_argument('--delay', type=float, default=0.15,
                       help='Delay between steps in seconds (for visualization)')
    parser.add_argument('--fps', type=int, default=None,
                       help='Override FPS for visualization')
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("CUDA not available, using CPU")
    else:
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available, falling back to CPU")
            device = 'cpu'
    
    print(f"Evaluation device: {device}")
    
    # Create environment
    render_mode = None if args.no_render else 'human'
    env = PokemonShowdownEnv(level=args.level, render_mode=render_mode)
    
    # Override FPS if specified
    if render_mode == 'human' and args.fps is not None:
        if env.renderer is not None:
            env.renderer.fps = args.fps
    elif render_mode == 'human':
        # Set to visualization FPS for smooth beam animation
        if env.renderer is None:
            from visualizer import PokemonRenderer
            env.renderer = PokemonRenderer(env.grid_size)
        env.renderer.fps = RENDER_CONFIG['fps']
    
    # Load models
    models = {}
    
    for agent in env.possible_agents:
        sample_obs = env.observation_space(agent).shape[0]
        model = ActorCritic(sample_obs).to(device)
        
        # Try to load checkpoint for this agent
        checkpoint_path = f"{args.checkpoint_dir}/checkpoint_level_{args.level}_agent_{agent}.pt"
        
        try:
            # For simplicity, try loading any checkpoint
            import glob
            import re
            
            checkpoints = glob.glob(f"{args.checkpoint_dir}/checkpoint_level_{args.level}_*.pt")
            if checkpoints:
                # Sort by step number to get latest
                def get_step(filename):
                    match = re.search(r'step_(\d+)', filename)
                    return int(match.group(1)) if match else 0
                
                checkpoint_path = max(checkpoints, key=get_step)
                print(f"Loading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                print(f"Warning: No checkpoint found for {agent}, using random policy")
        except Exception as e:
            print(f"Warning: Could not load checkpoint for {agent}: {e}")
            print("Using random policy")
        
        models[agent] = model
    
    # Create evaluator
    evaluator = Evaluator(env, models, device=device)
    
    # Run evaluation
    print(f"\nEvaluating on {args.level} for {args.num_episodes} episodes...")
    print(f"Render mode: {render_mode}")
    if render_mode == 'human':
        print(f"FPS: {env.renderer.fps if env.renderer else 'N/A'}")
        print(f"Delay between steps: {args.delay}s")
    
    stats = evaluator.evaluate(
        num_episodes=args.num_episodes,
        render=not args.no_render,
        deterministic=args.deterministic,
        delay=args.delay
    )
    
    # Print results
    print_evaluation_stats(stats)
    
    env.close()


if __name__ == '__main__':
    main()