"""
test_env.py - Script to visualize the Pokemon RL environment
"""

import numpy as np
import time
from environment.pokenv import PokemonRL


def random_policy(env, agent):
    """Random action selection"""
    return env.action_space(agent).sample()


def smart_random_policy(env, agent):
    """Slightly smarter policy - attacks more often"""
    state = env.agent_states[agent]
    target = state["target"]
    
    if target and not env.terminations.get(target, False):
        target_pos = env.agent_states[target]["pos"]
        my_pos = state["pos"]
        
        # Calculate Manhattan distance
        distance = abs(my_pos[0] - target_pos[0]) + abs(my_pos[1] - target_pos[1])
        
        # If in range, attack with 70% probability
        if distance <= env.obs_radius:
            if np.random.random() < 0.7:
                return 4  # Attack
        else:
            # Move towards target
            if my_pos[0] < target_pos[0]:
                return 1  # Down
            elif my_pos[0] > target_pos[0]:
                return 0  # Up
            elif my_pos[1] < target_pos[1]:
                return 3  # Right
            elif my_pos[1] > target_pos[1]:
                return 2  # Left
    
    # Default: random action
    return env.action_space(agent).sample()


def run_visualization(
    num_episodes=5,
    pokemon_list=None,
    grid_size=10,
    obs_radius=2,
    policy="smart_random",
    fps=5
):
    """
    Run environment visualization
    
    Args:
        num_episodes: Number of episodes to run
        pokemon_list: List of pokemon names (e.g., ["charmander", "bulbasaur", "squirtle"])
        grid_size: Grid size
        obs_radius: Observation radius
        policy: "random" or "smart_random"
        fps: Frames per second for rendering
    """
    
    # Create environment with visualization
    env = PokemonRL(
        pokemon_list=pokemon_list,
        grid_size=grid_size,
        obs_radius=obs_radius,
        num_obstacles=5,
        render_mode="human"
    )
    
    # Override FPS
    env.metadata["render_fps"] = fps
    
    # Select policy
    if policy == "random":
        policy_fn = random_policy
    else:
        policy_fn = smart_random_policy
    
    print(f"Running {num_episodes} episodes with {len(env.possible_agents)} agents")
    print(f"Agents: {env.possible_agents}")
    print(f"Grid Size: {grid_size}x{grid_size}, Obs Radius: {obs_radius}")
    print("-" * 60)
    
    for episode in range(num_episodes):
        observations, infos = env.reset()
        episode_rewards = {agent: 0 for agent in env.possible_agents}
        steps = 0
        
        print(f"\n[Episode {episode + 1}/{num_episodes}]")
        
        while env.agents:
            # Render
            env.render()
            
            # Get actions for all agents
            actions = {
                agent: policy_fn(env, agent)
                for agent in env.agents
            }
            
            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Accumulate rewards
            for agent, reward in rewards.items():
                episode_rewards[agent] += reward
            
            steps += 1
            
            # Small delay for better visualization
            time.sleep(0.1)
        
        # Print episode results
        print(f"  Steps: {steps}")
        print(f"  Rewards:")
        for agent, total_reward in episode_rewards.items():
            print(f"    {agent}: {total_reward:.2f}")
        
        # Find winner (highest reward)
        winner = max(episode_rewards, key=episode_rewards.get)
        print(f"  Winner: {winner} 🏆")
        
        # Wait before next episode
        time.sleep(2)
    
    env.close()
    print("\n" + "=" * 60)
    print("Visualization complete!")


if __name__ == "__main__":
    # Example 1: Classic trio
    print("=" * 60)
    print("SCENARIO 1: Classic Trio (Charmander, Bulbasaur, Squirtle)")
    print("=" * 60)
    run_visualization(
        num_episodes=3,
        pokemon_list=None,  # Default trio
        grid_size=10,
        obs_radius=5,
        policy="smart_random",
        fps=20
    )
    
    # Example 2: Different Pokemon
    print("\n" + "=" * 60)
    print("SCENARIO 2: Electric Battle (Pikachu, Geodude, Pidgey)")
    print("=" * 60)
    run_visualization(
        num_episodes=2,
        pokemon_list=["pikachu", "geodude", "pidgey"],
        grid_size=12,
        obs_radius=5,
        policy="smart_random",
        fps=8
    )
    
    # Example 3: Big battle
    print("\n" + "=" * 60)
    print("SCENARIO 3: Large Battle (6 Pokemon)")
    print("=" * 60)
    run_visualization(
        num_episodes=1,
        pokemon_list=["charmander", "bulbasaur", "squirtle", "pikachu", "machop", "onix"],
        grid_size=15,
        obs_radius=5,
        policy="smart_random",
        fps=10
    )