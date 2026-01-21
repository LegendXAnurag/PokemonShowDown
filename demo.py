"""
Demo script to showcase the Pokemon battle environment.
Runs a quick demo of different training levels.
"""

from pokemon_env import PokemonBattleEnv
import random


def demo_level(level, num_steps=20):
    """
    Run a quick demo of a specific training level.
    
    Args:
        level: Training level (1-6)
        num_steps: Number of steps to simulate
    """
    print("\n" + "=" * 60)
    print(f"DEMO: Training Level {level}")
    print("=" * 60)
    
    # Create environment
    env = PokemonBattleEnv(level=level)
    observations, infos = env.reset()
    
    print(f"\nConfiguration:")
    print(f"  Grid Size: {env.grid_size}x{env.grid_size}")
    print(f"  Obstacles: {len(env.obstacles)}")
    print(f"  Pokemon: {env.num_pokemon}")
    print(f"  Team Mode: {env.team_mode}")
    
    print(f"\nPokemon Lineup:")
    for agent, pokemon in env.pokemons.items():
        team_info = f" (Team {pokemon.team_id})" if env.team_mode else ""
        print(f"  {agent}: {pokemon.name} ({pokemon.poke_type}){team_info}")
    
    print("\nInitial State:")
    env._render_human()
    
    # Run simulation
    total_rewards = {agent: 0 for agent in env.agents}
    
    for step in range(num_steps):
        # Random actions (in actual training, these would come from the policy network)
        actions = {agent: random.randint(0, 7) for agent in env.agents}
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Accumulate rewards
        for agent in rewards:
            if agent in total_rewards:
                total_rewards[agent] += rewards[agent]
        
        # Check if battle ended
        if all(terminations.values()):
            print(f"\n*** Battle ended at step {step + 1} ***")
            break
    
    print("\nFinal State:")
    env._render_human()
    
    print("\nTotal Rewards:")
    for agent, reward in total_rewards.items():
        status = "ALIVE" if env.pokemons[agent].is_alive else "DEFEATED"
        print(f"  {agent}: {reward:6.2f} ({status})")
    
    print(f"\nLevel {level} Demo Complete!")


def main():
    """Run demos for all training levels."""
    print("\n" + "=" * 60)
    print("Pokemon ShowDown - Multi-Agent RL Co-Learning System")
    print("=" * 60)
    print("\nThis demo showcases the 6 progressive training levels")
    print("for Pokemon battle reinforcement learning.")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Demo each level
    demo_level(1, num_steps=15)
    demo_level(2, num_steps=15)
    demo_level(3, num_steps=20)
    demo_level(4, num_steps=25)
    demo_level(5, num_steps=30)
    demo_level(6, num_steps=25)
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("  ✓ 6 progressive training levels")
    print("  ✓ Type-based damage multipliers")
    print("  ✓ Beam-style line attacks")
    print("  ✓ Obstacle placement (levels 3-6)")
    print("  ✓ Multi-agent battles (up to 5 Pokemon)")
    print("  ✓ Team-based combat (level 6)")
    print("  ✓ Reward/penalty system")
    print("\nTo train agents, run: python trainer.py")
    print("For testing, run: python test_environment.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
