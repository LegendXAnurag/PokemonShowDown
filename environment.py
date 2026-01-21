"""
PettingZoo-based Pokemon battle environment for RL training.
Implements 6 training levels with increasing complexity.
"""

import numpy as np
import random
from typing import Optional, Dict, List, Tuple
from pettingzoo import ParallelEnv
from gymnasium import spaces
from pokemon import Pokemon
from pokemon_data import POKEMON_LIST


class PokemonBattleEnv(ParallelEnv):
    """
    Pokemon battle environment supporting multiple training levels.
    
    Training Levels:
    1. 5x5 grid, no obstacles, 1v1
    2. 6x6 grid, no obstacles, 1v1
    3. 6x6 grid, 2-3 obstacles, 1v1
    4. 10x10 grid, 4-5 obstacles, 1v1v1
    5. 10x10 grid, 4-5 obstacles, 1v1v1v1v1
    6. 10x10 grid, 4-5 obstacles, 2v2 (team battles)
    """
    
    metadata = {
        "name": "pokemon_battle_v0",
        "render_modes": ["human", "rgb_array"],
    }
    
    def __init__(self, level=1, render_mode=None, max_steps=500):
        """
        Initialize the Pokemon battle environment.
        
        Args:
            level: Training level (1-6)
            render_mode: Rendering mode ("human" or "rgb_array")
            max_steps: Maximum steps per episode
        """
        super().__init__()
        
        self.level = level
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        
        # Set grid size based on level
        if level == 1:
            self.grid_size = 5
            self.num_obstacles = 0
            self.num_pokemon = 2
            self.team_mode = False
        elif level == 2:
            self.grid_size = 6
            self.num_obstacles = 0
            self.num_pokemon = 2
            self.team_mode = False
        elif level == 3:
            self.grid_size = 6
            self.num_obstacles = random.randint(2, 3)
            self.num_pokemon = 2
            self.team_mode = False
        elif level == 4:
            self.grid_size = 10
            self.num_obstacles = random.randint(4, 5)
            self.num_pokemon = 3
            self.team_mode = False
        elif level == 5:
            self.grid_size = 10
            self.num_obstacles = random.randint(4, 5)
            self.num_pokemon = 5
            self.team_mode = False
        elif level == 6:
            self.grid_size = 10
            self.num_obstacles = random.randint(4, 5)
            self.num_pokemon = 4
            self.team_mode = True
        else:
            raise ValueError(f"Invalid level: {level}. Must be 1-6.")
            
        # Action space: 0=Up, 1=Down, 2=Left, 3=Right, 4=Attack Up, 5=Attack Down, 6=Attack Left, 7=Attack Right
        self._action_spaces = {}
        self._observation_spaces = {}
        
        # Initialize grid and Pokemon
        self.grid = None
        self.obstacles = []
        self.pokemons = {}
        self.agents = []
        self.possible_agents = []
        
        # Rewards configuration
        self.reward_defeat_opponent = 100
        self.reward_damage_opponent = 10
        self.penalty_take_damage = -5
        self.penalty_defeated = -50
        self.penalty_time_step = -0.1
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Returns:
            observations: Dictionary of observations for each agent
            infos: Dictionary of info for each agent
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.current_step = 0
        
        # Initialize grid (0 = empty, 1 = obstacle)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Place obstacles (ensuring they don't share edges or vertices)
        self.obstacles = []
        self._place_obstacles()
        
        # Select and initialize Pokemon
        self._initialize_pokemon()
        
        # Generate agents list
        self.agents = [f"pokemon_{i}" for i in range(self.num_pokemon)]
        self.possible_agents = self.agents.copy()
        
        # Define action and observation spaces for each agent
        for agent in self.agents:
            # Action space: 8 actions (4 moves + 4 attacks)
            self._action_spaces[agent] = spaces.Discrete(8)
            
            # Observation space: grid + pokemon info
            # Grid: grid_size x grid_size (0=empty, 1=obstacle, 2=self, 3=teammate, 4=opponent)
            # Pokemon info: health, position, type (one-hot encoded)
            obs_size = self.grid_size * self.grid_size + 10 + len(self.pokemons) * 3
            self._observation_spaces[agent] = spaces.Box(
                low=0, high=1, shape=(obs_size,), dtype=np.float32
            )
        
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
        
    def _place_obstacles(self):
        """Place obstacles on the grid ensuring they don't share edges or vertices."""
        placed = 0
        max_attempts = 1000
        attempts = 0
        
        while placed < self.num_obstacles and attempts < max_attempts:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            
            # Check if position is valid (not too close to other obstacles)
            valid = True
            for ox, oy in self.obstacles:
                # Check if shares edge or vertex
                if abs(x - ox) <= 1 and abs(y - oy) <= 1:
                    valid = False
                    break
                    
            if valid:
                self.obstacles.append((x, y))
                self.grid[y, x] = 1
                placed += 1
                
            attempts += 1
            
    def _initialize_pokemon(self):
        """Initialize Pokemon for the battle."""
        # Select random Pokemon
        selected_pokemon = random.sample(POKEMON_LIST, self.num_pokemon)
        
        self.pokemons = {}
        for i, poke_data in enumerate(selected_pokemon):
            pokemon = Pokemon(
                name=poke_data['name'],
                poke_type=poke_data['type'],
                sprite_path=poke_data['sprite'],
                base_health=poke_data['base_health'],
                base_attack=poke_data['base_attack']
            )
            
            # Place Pokemon on grid
            pos = self._get_random_empty_position()
            pokemon.set_position(*pos)
            
            # Set team for team mode
            if self.team_mode:
                pokemon.set_team(i % 2)  # Alternate teams
                
            self.pokemons[f"pokemon_{i}"] = pokemon
            
    def _get_random_empty_position(self):
        """Get a random empty position on the grid."""
        while True:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            
            # Check if position is empty
            if self.grid[y, x] == 0:
                # Check if no Pokemon is at this position
                occupied = False
                for pokemon in self.pokemons.values():
                    if pokemon.position == (x, y):
                        occupied = True
                        break
                        
                if not occupied:
                    return (x, y)
                    
    def step(self, actions: Dict[str, int]):
        """
        Execute one step in the environment.
        
        Args:
            actions: Dictionary of actions for each agent
            
        Returns:
            observations: Dictionary of observations for each agent
            rewards: Dictionary of rewards for each agent
            terminations: Dictionary of termination flags for each agent
            truncations: Dictionary of truncation flags for each agent
            infos: Dictionary of info for each agent
        """
        self.current_step += 1
        
        rewards = {agent: self.penalty_time_step for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        # Execute actions for all agents
        for agent, action in actions.items():
            if agent not in self.pokemons or not self.pokemons[agent].is_alive:
                continue
                
            pokemon = self.pokemons[agent]
            
            # Movement actions (0-3)
            if action < 4:
                self._execute_move(pokemon, action)
            # Attack actions (4-7)
            else:
                damage_dealt = self._execute_attack(pokemon, action - 4, agent, rewards)
                
        # Check for termination conditions
        alive_agents = [agent for agent in self.agents if self.pokemons[agent].is_alive]
        
        if self.team_mode:
            # Check if one team is eliminated
            teams_alive = set(self.pokemons[agent].team_id for agent in alive_agents)
            if len(teams_alive) <= 1:
                for agent in self.agents:
                    terminations[agent] = True
        else:
            # Check if only one Pokemon remains
            if len(alive_agents) <= 1:
                for agent in self.agents:
                    terminations[agent] = True
                    
        # Check for max steps
        if self.current_step >= self.max_steps:
            for agent in self.agents:
                truncations[agent] = True
                
        # Update agents list (remove defeated Pokemon)
        self.agents = [agent for agent in self.agents if self.pokemons[agent].is_alive]
        
        observations = self._get_observations()
        
        return observations, rewards, terminations, truncations, infos
        
    def _execute_move(self, pokemon: Pokemon, direction: int):
        """Execute a movement action."""
        x, y = pokemon.position
        
        # Calculate new position
        if direction == 0:  # Up
            new_y = max(0, y - 1)
            new_x = x
        elif direction == 1:  # Down
            new_y = min(self.grid_size - 1, y + 1)
            new_x = x
        elif direction == 2:  # Left
            new_x = max(0, x - 1)
            new_y = y
        else:  # Right
            new_x = min(self.grid_size - 1, x + 1)
            new_y = y
            
        # Check if new position is valid (not obstacle, not occupied)
        if self.grid[new_y, new_x] == 0:
            occupied = False
            for other_pokemon in self.pokemons.values():
                if other_pokemon.position == (new_x, new_y) and other_pokemon != pokemon:
                    occupied = True
                    break
                    
            if not occupied:
                pokemon.set_position(new_x, new_y)
                
    def _execute_attack(self, pokemon: Pokemon, direction: int, agent: str, rewards: Dict):
        """Execute an attack action (beam attack in a line)."""
        x, y = pokemon.position
        
        # Determine attack direction
        if direction == 0:  # Up
            dx, dy = 0, -1
        elif direction == 1:  # Down
            dx, dy = 0, 1
        elif direction == 2:  # Left
            dx, dy = -1, 0
        else:  # Right
            dx, dy = 1, 0
            
        # Trace the beam until it hits an obstacle or Pokemon
        current_x, current_y = x + dx, y + dy
        
        while (0 <= current_x < self.grid_size and 0 <= current_y < self.grid_size):
            # Check if beam hits an obstacle
            if self.grid[current_y, current_x] == 1:
                break
                
            # Check if beam hits a Pokemon
            for other_agent, other_pokemon in self.pokemons.items():
                if other_pokemon.position == (current_x, current_y) and other_pokemon.is_alive:
                    # Check if it's a teammate in team mode
                    if self.team_mode and pokemon.team_id == other_pokemon.team_id:
                        continue
                        
                    # Deal damage
                    damage = other_pokemon.take_damage(pokemon.attack_damage(), pokemon.poke_type)
                    
                    # Update rewards
                    rewards[agent] += self.reward_damage_opponent
                    rewards[other_agent] += self.penalty_take_damage
                    
                    if not other_pokemon.is_alive:
                        rewards[agent] += self.reward_defeat_opponent
                        rewards[other_agent] += self.penalty_defeated
                        
                    return damage
                    
            # Continue beam
            current_x += dx
            current_y += dy
            
        return 0
        
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all agents."""
        observations = {}
        
        for agent in self.agents:
            if agent not in self.pokemons:
                continue
                
            pokemon = self.pokemons[agent]
            
            # Create grid observation
            grid_obs = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            
            # Mark obstacles
            grid_obs[self.grid == 1] = 1
            
            # Mark Pokemon positions
            for other_agent, other_pokemon in self.pokemons.items():
                if not other_pokemon.is_alive:
                    continue
                    
                px, py = other_pokemon.position
                if other_agent == agent:
                    grid_obs[py, px] = 2  # Self
                elif self.team_mode and pokemon.team_id == other_pokemon.team_id:
                    grid_obs[py, px] = 3  # Teammate
                else:
                    grid_obs[py, px] = 4  # Opponent
                    
            # Flatten grid
            grid_flat = grid_obs.flatten()
            
            # Add Pokemon info
            pokemon_info = [
                pokemon.health / pokemon.base_health,  # Normalized health
                pokemon.position[0] / self.grid_size,  # Normalized x position
                pokemon.position[1] / self.grid_size,  # Normalized y position
            ]
            
            # Add type as one-hot encoding (8 types)
            type_encoding = [0] * 8
            type_map = {'Fire': 0, 'Water': 1, 'Grass': 2, 'Electric': 3, 'Ice': 4, 'Fighting': 5, 'Psychic': 6, 'Dragon': 7}
            if pokemon.poke_type in type_map:
                type_encoding[type_map[pokemon.poke_type]] = 1
            pokemon_info.extend(type_encoding)
            
            # Add info about other Pokemon
            for other_agent in sorted(self.pokemons.keys()):
                if other_agent == agent:
                    continue
                other_pokemon = self.pokemons[other_agent]
                if other_pokemon.is_alive:
                    pokemon_info.extend([
                        other_pokemon.health / other_pokemon.base_health,
                        (other_pokemon.position[0] - pokemon.position[0]) / self.grid_size,
                        (other_pokemon.position[1] - pokemon.position[1]) / self.grid_size,
                    ])
                else:
                    pokemon_info.extend([0, 0, 0])
                    
            # Combine observations
            obs = np.concatenate([grid_flat, pokemon_info])
            
            # Pad or truncate to match observation space
            expected_size = self._observation_spaces[agent].shape[0]
            if len(obs) < expected_size:
                obs = np.pad(obs, (0, expected_size - len(obs)))
            elif len(obs) > expected_size:
                obs = obs[:expected_size]
                
            observations[agent] = obs.astype(np.float32)
            
        return observations
        
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
            
    def _render_human(self):
        """Render environment to console."""
        print(f"\n=== Step {self.current_step} ===")
        print(f"Level: {self.level}, Grid: {self.grid_size}x{self.grid_size}")
        
        # Create display grid
        display = [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Mark obstacles
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.grid[y, x] == 1:
                    display[y][x] = '#'
                    
        # Mark Pokemon
        for i, (agent, pokemon) in enumerate(self.pokemons.items()):
            if pokemon.is_alive:
                px, py = pokemon.position
                display[py][px] = str(i)
                
        # Print grid
        print("+" + "-" * self.grid_size + "+")
        for row in display:
            print("|" + "".join(row) + "|")
        print("+" + "-" * self.grid_size + "+")
        
        # Print Pokemon status
        for agent, pokemon in self.pokemons.items():
            status = "ALIVE" if pokemon.is_alive else "DEFEATED"
            print(f"{agent}: {pokemon.name} ({pokemon.poke_type}) HP:{pokemon.health:.1f} {status}")
            
    def _render_rgb_array(self):
        """Render environment as RGB array (for pygame rendering)."""
        # This would return an RGB array for pygame rendering
        # Implementation would depend on pygame integration
        pass
        
    def observation_space(self, agent):
        """Return observation space for an agent."""
        return self._observation_spaces[agent]
        
    def action_space(self, agent):
        """Return action space for an agent."""
        return self._action_spaces[agent]
