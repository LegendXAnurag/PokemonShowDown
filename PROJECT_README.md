# Pokemon ShowDown - Minimalistic RL Co-Learning

A reinforcement learning project implementing multi-agent co-learning using Pokemon battles. Agents learn to battle against each other across 6 progressive training levels.

## Features

- **Multi-Agent RL**: Multiple agents influence each other's learning
- **Progressive Training**: 6 levels with increasing complexity
- **Type-Based Combat**: Pokemon type advantages/disadvantages affect damage
- **Team Battles**: Advanced levels support team-based combat
- **Visual Feedback**: Pygame-based visualization of battles

## Pokemon Types

The system includes 8 Pokemon types with different type advantages:
- Fire (Charizard)
- Water (Blastoise)
- Grass (Venusaur)
- Electric (Pikachu)
- Ice (Articuno)
- Fighting (Machamp)
- Psychic (Mewtwo)
- Dragon (Dragonite)

## Training Levels

### Level 1
- 5x5 grid
- No obstacles
- 1v1 Pokemon battle

### Level 2
- 6x6 grid
- No obstacles
- 1v1 Pokemon battle

### Level 3
- 6x6 grid
- 2-3 obstacles
- 1v1 Pokemon battle

### Level 4
- 10x10 grid
- 4-5 obstacles
- 1v1v1 free-for-all (3 Pokemon)

### Level 5
- 10x10 grid
- 4-5 obstacles
- 1v1v1v1v1 free-for-all (5 Pokemon)

### Level 6
- 10x10 grid
- 4-5 obstacles
- 2v2 team battles (4 Pokemon)
- Agents learn not to attack teammates

## Game Mechanics

### Action Space
- 0-3: Movement (Up, Down, Left, Right)
- 4-7: Attack (Up, Down, Left, Right)

### Attack System
- Beam-style attacks in a straight line
- Damages first Pokemon in the line of fire
- Blocked by obstacles

### Damage System
- Base attack and health are equal for all Pokemon
- Damage multipliers based on type effectiveness:
  - Super effective: 2.0x damage
  - Normal: 1.0x damage
  - Not very effective: 0.5x damage

### Rewards
- Defeating opponent: +100
- Damaging opponent: +10
- Taking damage: -5
- Getting defeated: -50
- Time elapsed: -0.1 (encourages active combat)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train agents on all 6 levels:

```python
from trainer import train_all_levels

train_all_levels(num_episodes_per_level=1000)
```

### Single Level Training

```python
from pokemon_env import PokemonBattleEnv
from trainer import RandomPolicyTrainer

# Create environment for level 3
env = PokemonBattleEnv(level=3)

# Create and train
trainer = RandomPolicyTrainer(env, level=3, entropy_coef=0.05)
trainer.train(num_episodes=1000)
```

### Visualization

```python
from pokemon_env import PokemonBattleEnv
from visualizer import PokemonVisualizer

env = PokemonBattleEnv(level=1)
visualizer = PokemonVisualizer(grid_size=env.grid_size)

observations, infos = env.reset()

# Game loop
for step in range(100):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    visualizer.render(env, step)
    
    if all(terminations.values()):
        break

visualizer.close()
```

## Architecture

### Core Components

1. **Pokemon Class** (`pokemon.py`): Represents individual Pokemon with attributes and state
2. **Environment** (`pokemon_env.py`): PettingZoo-based parallel environment
3. **Visualizer** (`visualizer.py`): Pygame-based rendering system
4. **Trainer** (`trainer.py`): Training infrastructure using TorchRL
5. **Pokemon Data** (`pokemon_data.py`): Static data for types and effectiveness

### Technology Stack

- **Environment**: PettingZoo (parallel multi-agent environment)
- **Rendering**: Pygame
- **RL Framework**: TorchRL (PPO/MAPPO/IPPO algorithms)
- **Deep Learning**: PyTorch

## Training Philosophy

The training emphasizes:
- **Exploration**: High entropy coefficient ensures agents explore all actions
- **Type Strategy**: Agents learn to dodge when at type disadvantage
- **Co-Learning**: Multiple agents learn simultaneously, adapting to each other's strategies
- **Progressive Complexity**: Start simple, gradually increase difficulty

## Future Enhancements

- Full MAPPO/IPPO implementation with TorchRL
- Advanced neural network architectures
- Curriculum learning across levels
- Model checkpointing and loading
- Performance metrics and analysis
- Additional Pokemon types and moves

## License

See LICENSE file for details.
