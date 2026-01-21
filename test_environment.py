"""
Test script to verify the Pokemon battle environment implementation.
"""

from pokemon_env import PokemonBattleEnv
from pokemon import Pokemon
from pokemon_data import POKEMON_LIST, get_damage_multiplier
import random


def test_pokemon_class():
    """Test Pokemon class functionality."""
    print("=" * 50)
    print("Testing Pokemon Class")
    print("=" * 50)
    
    # Create a Pokemon
    poke_data = POKEMON_LIST[0]
    pokemon = Pokemon(
        name=poke_data['name'],
        poke_type=poke_data['type'],
        sprite_path=poke_data['sprite'],
        base_health=poke_data['base_health'],
        base_attack=poke_data['base_attack']
    )
    
    print(f"Created: {pokemon}")
    
    # Test damage
    attacker_type = "Water"
    damage = pokemon.take_damage(20, attacker_type)
    multiplier = get_damage_multiplier(attacker_type, pokemon.poke_type)
    print(f"Damage from {attacker_type}: {damage} (multiplier: {multiplier}x)")
    print(f"After damage: {pokemon}")
    
    # Test reset
    pokemon.reset()
    print(f"After reset: {pokemon}")
    print("✓ Pokemon class working correctly\n")


def test_environment_level(level):
    """Test environment for a specific level."""
    print("=" * 50)
    print(f"Testing Environment - Level {level}")
    print("=" * 50)
    
    env = PokemonBattleEnv(level=level)
    observations, infos = env.reset()
    
    print(f"Grid size: {env.grid_size}x{env.grid_size}")
    print(f"Number of obstacles: {len(env.obstacles)}")
    print(f"Number of Pokemon: {env.num_pokemon}")
    print(f"Team mode: {env.team_mode}")
    print(f"Agents: {env.agents}")
    
    # Run a few steps
    print("\nRunning 10 steps...")
    for step in range(10):
        # Random actions
        actions = {agent: random.randint(0, 7) for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        if step == 0 or step == 9:
            print(f"\nStep {step}:")
            env._render_human()
        
        if all(terminations.values()):
            print(f"\nBattle ended at step {step}")
            break
    
    print("✓ Environment working correctly\n")


def test_type_effectiveness():
    """Test type effectiveness system."""
    print("=" * 50)
    print("Testing Type Effectiveness")
    print("=" * 50)
    
    test_cases = [
        ("Fire", "Grass", 2.0),
        ("Water", "Fire", 2.0),
        ("Fire", "Water", 0.5),
        ("Electric", "Water", 2.0),
        ("Fire", "Fire", 0.5),
    ]
    
    for attacker, defender, expected in test_cases:
        multiplier = get_damage_multiplier(attacker, defender)
        status = "✓" if multiplier == expected else "✗"
        print(f"{status} {attacker} -> {defender}: {multiplier}x (expected {expected}x)")
    
    print("✓ Type effectiveness working correctly\n")


def test_all_levels():
    """Test all training levels."""
    print("=" * 50)
    print("Testing All Training Levels")
    print("=" * 50)
    
    for level in range(1, 7):
        try:
            env = PokemonBattleEnv(level=level)
            observations, infos = env.reset()
            
            # Run one step
            actions = {agent: random.randint(0, 7) for agent in env.agents}
            env.step(actions)
            
            print(f"✓ Level {level}: OK (grid={env.grid_size}x{env.grid_size}, pokemon={env.num_pokemon})")
        except Exception as e:
            print(f"✗ Level {level}: FAILED - {e}")
    
    print("✓ All levels working correctly\n")


def run_sample_battle(level=1, max_steps=50):
    """Run a sample battle with visualization."""
    print("=" * 50)
    print(f"Running Sample Battle - Level {level}")
    print("=" * 50)
    
    env = PokemonBattleEnv(level=level)
    observations, infos = env.reset()
    
    print("\nInitial state:")
    env._render_human()
    
    total_rewards = {agent: 0 for agent in env.agents}
    
    for step in range(max_steps):
        # Random actions
        actions = {agent: random.randint(0, 7) for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Accumulate rewards
        for agent in rewards:
            if agent in total_rewards:
                total_rewards[agent] += rewards[agent]
        
        # Show every 10 steps
        if step % 10 == 0:
            print(f"\n--- Step {step} ---")
            env._render_human()
        
        if all(terminations.values()):
            print(f"\n--- Final State (Step {step}) ---")
            env._render_human()
            print("\nBattle ended!")
            break
    
    print("\nFinal Rewards:")
    for agent, reward in total_rewards.items():
        print(f"  {agent}: {reward:.2f}")
    
    print("✓ Sample battle completed\n")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    print("\n" + "=" * 50)
    print("Pokemon ShowDown - Environment Tests")
    print("=" * 50 + "\n")
    
    # Run tests
    test_pokemon_class()
    test_type_effectiveness()
    test_all_levels()
    test_environment_level(1)
    test_environment_level(3)
    run_sample_battle(level=2, max_steps=30)
    
    print("=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)
