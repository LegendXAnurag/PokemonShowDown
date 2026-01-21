import functools
import numpy as np
import pygame
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv
from typing import List, Dict, Optional
from agent.pokemon import Pokemon, PokemonType, get_pokemon, create_balanced_team


class PokemonRL(ParallelEnv):
    metadata = {
        "name": "pokemon_rl_v0",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self, 
        pokemon_list: Optional[List[str]] = None,
        grid_size: int = 10, 
        obs_radius: int = 2, 
        num_obstacles: int = 5, 
        render_mode: Optional[str] = None
    ):
        """
        Initialize Pokemon RL environment
        
        Args:
            pokemon_list: List of pokemon names to use (e.g., ["charmander", "bulbasaur", "squirtle"])
                         If None, uses default balanced team
            grid_size: Size of the grid (n x n)
            obs_radius: Observation radius (agent sees (2*radius+1) x (2*radius+1) grid)
            num_obstacles: Number of obstacles on the map
            render_mode: "human" or "rgb_array"
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.obs_radius = obs_radius
        self.num_obstacles = num_obstacles
        self.render_mode = render_mode
        
        # Create Pokemon instances
        if pokemon_list is None:
            # Default: classic trio
            self.pokemon_team = create_balanced_team(3)
        else:
            self.pokemon_team = [get_pokemon(name) for name in pokemon_list]
        
        # Create agent names
        self.possible_agents = [
            f"{pokemon.name.lower()}_{i}" 
            for i, pokemon in enumerate(self.pokemon_team)
        ]
        
        # Map agent names to Pokemon
        self.agent_to_pokemon = {
            agent: pokemon 
            for agent, pokemon in zip(self.possible_agents, self.pokemon_team)
        }
        
        self.agent_name_mapping = {
            name: i for i, name in enumerate(self.possible_agents)
        }
        
        # Get all unique types in the game
        self.types_in_game = list(set(p.poke_type for p in self.pokemon_team))
        self.type_to_channel = {ptype: i for i, ptype in enumerate(self.types_in_game)}
        
        # Action space: 0=Up, 1=Down, 2=Left, 3=Right, 4=Attack, 5=Switch Target
        self._action_spaces = {agent: Discrete(6) for agent in self.possible_agents}
        
        # Observation space
        # Channels: [Empty, Obstacle] + [Type_1, Type_2, ...] for each type in game
        # Plus: own_hp (1), enemy_hps (n-1), target_index (1)
        obs_grid_size = (2 * obs_radius + 1) ** 2
        num_type_channels = len(self.types_in_game)
        num_enemies = len(self.possible_agents) - 1
        
        obs_size = obs_grid_size * (2 + num_type_channels) + 2 + num_enemies
        
        self._observation_spaces = {
            agent: Box(
                low=0.0,
                high=1.0,
                shape=(obs_size,),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }
        
        # Pygame initialization
        self.window = None
        self.clock = None
        self.cell_size = 60
        self.window_size = self.grid_size * self.cell_size

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.agents = self.possible_agents[:]
        self.timestep = 0
        
        # Initialize grid (0=empty, 1=obstacle)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # Place obstacles randomly
        obstacle_positions = set()
        while len(obstacle_positions) < self.num_obstacles:
            pos = (np.random.randint(1, self.grid_size - 1), 
                   np.random.randint(1, self.grid_size - 1))
            obstacle_positions.add(pos)
        
        for pos in obstacle_positions:
            self.grid[pos[0], pos[1]] = 1
        
        # Initialize agent states
        self.agent_states = {}
        occupied = obstacle_positions.copy()
        
        for agent in self.agents:
            pokemon = self.agent_to_pokemon[agent]
            
            # Spawn near edges
            while True:
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(0, self.grid_size)
                
                # Prefer edge spawns
                if (x < 2 or x >= self.grid_size - 2 or 
                    y < 2 or y >= self.grid_size - 2):
                    if (x, y) not in occupied:
                        occupied.add((x, y))
                        break
            
            # Find initial target (first other agent)
            other_agents = [a for a in self.agents if a != agent]
            initial_target = other_agents[0] if other_agents else None
            
            self.agent_states[agent] = {
                "pos": np.array([x, y], dtype=np.int32),
                "hp": pokemon.max_hp,
                "max_hp": pokemon.max_hp,
                "target": initial_target,
                "alive": True,
            }
        
        # Initialize rewards and terminations
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        observations = self._get_observations()
        
        return observations, self.infos

    def step(self, actions):
        # Reset rewards for this step
        self.rewards = {agent: 0.0 for agent in self.agents}
        
        # 1. Process movement actions
        for agent in self.agents:
            if self.terminations[agent]:
                continue
            
            action = actions[agent]
            if action < 4:  # Movement action
                self._move_agent(agent, action)
            elif action == 5:  # Switch target
                self._switch_target(agent)
        
        # 2. Process attack actions
        for agent in self.agents:
            if self.terminations[agent]:
                continue
            
            action = actions[agent]
            if action == 4:  # Attack action
                self._process_attack(agent)
        
        # 3. Check for deaths and assign rewards
        for agent in self.agents:
            if self.agent_states[agent]["hp"] <= 0 and not self.terminations[agent]:
                self.terminations[agent] = True
                self.agent_states[agent]["alive"] = False
                
                # Death penalty
                self.rewards[agent] -= 100.0
                
                # Reward survivors
                for other_agent in self.agents:
                    if not self.terminations[other_agent]:
                        self.rewards[other_agent] += 50.0
        
        # 4. Check if game should end
        alive_count = sum(1 for agent in self.agents if not self.terminations[agent])
        
        if alive_count <= 1:
            # Game over - give winner bonus
            for agent in self.agents:
                if not self.terminations[agent]:
                    self.rewards[agent] += 100.0  # Victory bonus
                    self.terminations[agent] = True
                    self.truncations[agent] = True
        
        # Remove dead agents from active list
        self.agents = [agent for agent in self.agents if not self.terminations[agent]]
        
        self.timestep += 1
        
        # Get observations
        observations = self._get_observations()
        
        return observations, self.rewards, self.terminations, self.truncations, self.infos

    def _move_agent(self, agent, action):
        """Move agent based on action (0=Up, 1=Down, 2=Left, 3=Right)"""
        pos = self.agent_states[agent]["pos"]
        new_pos = pos.copy()
        
        if action == 0:  # Up
            new_pos[0] -= 1
        elif action == 1:  # Down
            new_pos[0] += 1
        elif action == 2:  # Left
            new_pos[1] -= 1
        elif action == 3:  # Right
            new_pos[1] += 1
        
        # Check bounds
        if (0 <= new_pos[0] < self.grid_size and 
            0 <= new_pos[1] < self.grid_size):
            
            # Check obstacle
            if self.grid[new_pos[0], new_pos[1]] == 0:
                
                # Check collision with other agents
                collision = False
                for other_agent in self.agents:
                    if other_agent != agent and not self.terminations[other_agent]:
                        other_pos = self.agent_states[other_agent]["pos"]
                        if np.array_equal(new_pos, other_pos):
                            collision = True
                            break
                
                if not collision:
                    self.agent_states[agent]["pos"] = new_pos

    def _switch_target(self, agent):
        """Switch to next available target"""
        current_target = self.agent_states[agent]["target"]
        
        # Get list of alive enemies
        alive_enemies = [a for a in self.possible_agents 
                        if a != agent and not self.terminations.get(a, False)]
        
        if not alive_enemies:
            self.agent_states[agent]["target"] = None
            return
        
        # Find current target index and switch to next
        if current_target in alive_enemies:
            current_idx = alive_enemies.index(current_target)
            next_idx = (current_idx + 1) % len(alive_enemies)
            self.agent_states[agent]["target"] = alive_enemies[next_idx]
        else:
            # Current target is dead or invalid, pick first alive enemy
            self.agent_states[agent]["target"] = alive_enemies[0]

    def _process_attack(self, attacker_agent):
        """Process attack from attacker to their target"""
        target_name = self.agent_states[attacker_agent]["target"]
        
        # Check if target is valid
        if not target_name or self.terminations.get(target_name, False):
            return
        
        attacker_pos = self.agent_states[attacker_agent]["pos"]
        target_pos = self.agent_states[target_name]["pos"]
        
        # Check if target is in range (Manhattan distance)
        distance = np.abs(attacker_pos[0] - target_pos[0]) + np.abs(attacker_pos[1] - target_pos[1])
        
        if distance <= self.obs_radius:
            # Get Pokemon instances
            attacker_pokemon = self.agent_to_pokemon[attacker_agent]
            target_pokemon = self.agent_to_pokemon[target_name]
            
            # Calculate damage using Pokemon's method
            damage = attacker_pokemon.calculate_damage(target_pokemon)
            
            # Apply damage
            self.agent_states[target_name]["hp"] -= damage
            
            # Reward for dealing damage
            self.rewards[attacker_agent] += damage * 0.5
            
            # Penalty for taking damage
            self.rewards[target_name] -= damage * 0.3

    def _get_observations(self):
        """Get observations for all agents"""
        observations = {}
        
        for agent in self.possible_agents:
            if self.terminations.get(agent, False):
                # Dead agents get zero observation
                obs_size = self._observation_spaces[agent].shape[0]
                observations[agent] = np.zeros(obs_size, dtype=np.float32)
            else:
                observations[agent] = self._get_agent_observation(agent)
        
        return observations

    def _get_agent_observation(self, agent):
        """Get observation for a specific agent"""
        pos = self.agent_states[agent]["pos"]
        x, y = pos[0], pos[1]
        
        # Create padded grid for easy slicing
        padded_size = self.grid_size + 2 * self.obs_radius
        padded_grid = np.ones((padded_size, padded_size), dtype=np.int8)  # Walls around
        padded_grid[self.obs_radius:self.obs_radius + self.grid_size,
                    self.obs_radius:self.obs_radius + self.grid_size] = self.grid
        
        # Slice the observable region
        obs_x = x + self.obs_radius
        obs_y = y + self.obs_radius
        obs_slice = padded_grid[obs_x - self.obs_radius:obs_x + self.obs_radius + 1,
                                obs_y - self.obs_radius:obs_y + self.obs_radius + 1]
        
        # Create observation grid: [Empty, Obstacle] + [Type_1, Type_2, ...]
        obs_size = 2 * self.obs_radius + 1
        num_channels = 2 + len(self.types_in_game)
        obs_grid = np.zeros((num_channels, obs_size, obs_size), dtype=np.float32)
        
        # Channel 0: Empty spaces
        obs_grid[0] = (obs_slice == 0).astype(np.float32)
        
        # Channel 1: Obstacles (including walls)
        obs_grid[1] = (obs_slice == 1).astype(np.float32)
        
        # Channels 2+: Agent types
        for other_agent in self.agents:
            if other_agent == agent or self.terminations.get(other_agent, False):
                continue
            
            other_pos = self.agent_states[other_agent]["pos"]
            rel_x = other_pos[0] - x + self.obs_radius
            rel_y = other_pos[1] - y + self.obs_radius
            
            # Check if in observable range
            if 0 <= rel_x < obs_size and 0 <= rel_y < obs_size:
                other_pokemon = self.agent_to_pokemon[other_agent]
                type_channel = 2 + self.type_to_channel[other_pokemon.poke_type]
                obs_grid[type_channel, rel_x, rel_y] = 1.0
        
        # Flatten the grid
        obs_flat = obs_grid.flatten()
        
        # Additional features
        own_hp = self.agent_states[agent]["hp"] / self.agent_states[agent]["max_hp"]
        
        # Get HPs of all enemies (in order)
        enemy_hps = []
        for other_agent in self.possible_agents:
            if other_agent != agent:
                if self.terminations.get(other_agent, False):
                    enemy_hps.append(0.0)
                else:
                    other_pos = self.agent_states[other_agent]["pos"]
                    distance = np.abs(pos[0] - other_pos[0]) + np.abs(pos[1] - other_pos[1])
                    if distance <= self.obs_radius:
                        enemy_hps.append(
                            self.agent_states[other_agent]["hp"] / 
                            self.agent_states[other_agent]["max_hp"]
                        )
                    else:
                        enemy_hps.append(0.0)  # Not in view
        
        # Target index (normalized)
        target_name = self.agent_states[agent]["target"]
        if target_name and target_name in self.possible_agents:
            target_idx = self.possible_agents.index(target_name)
            target_feature = target_idx / len(self.possible_agents)
        else:
            target_feature = 0.0
        
        # Concatenate all features
        additional_features = np.array([own_hp] + enemy_hps + [target_feature], dtype=np.float32)
        
        observation = np.concatenate([obs_flat, additional_features])
        
        return observation

    def render(self):
        if self.render_mode is None:
            return
        
        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
                pygame.display.set_caption("Pokemon RL")
            else:
                self.window = pygame.Surface((self.window_size, self.window_size))
            
            self.clock = pygame.time.Clock()
        
        # Colors
        bg_color = (240, 240, 240)
        grid_color = (200, 200, 200)
        obstacle_color = (64, 64, 64)
        target_line_color = (255, 215, 0)
        
        # Draw background
        self.window.fill(bg_color)
        
        # Draw grid lines
        for i in range(self.grid_size + 1):
            pygame.draw.line(
                self.window,
                grid_color,
                (i * self.cell_size, 0),
                (i * self.cell_size, self.window_size),
                1
            )
            pygame.draw.line(
                self.window,
                grid_color,
                (0, i * self.cell_size),
                (self.window_size, i * self.cell_size),
                1
            )
        
        # Draw obstacles
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.grid[x, y] == 1:
                    rect = pygame.Rect(
                        y * self.cell_size + 5,
                        x * self.cell_size + 5,
                        self.cell_size - 10,
                        self.cell_size - 10
                    )
                    pygame.draw.rect(self.window, obstacle_color, rect)
        
        # Draw agents
        for agent in self.possible_agents:
            if self.terminations.get(agent, False):
                continue
            
            pokemon = self.agent_to_pokemon[agent]
            pos = self.agent_states[agent]["pos"]
            color = pokemon.color
            
            # Draw agent as circle
            center = (
                pos[1] * self.cell_size + self.cell_size // 2,
                pos[0] * self.cell_size + self.cell_size // 2
            )
            pygame.draw.circle(self.window, color, center, self.cell_size // 3)
            
            # Draw HP bar
            hp_ratio = self.agent_states[agent]["hp"] / self.agent_states[agent]["max_hp"]
            bar_width = self.cell_size - 10
            bar_height = 5
            bar_x = pos[1] * self.cell_size + 5
            bar_y = pos[0] * self.cell_size + self.cell_size - 10
            
            # Background (red)
            pygame.draw.rect(
                self.window,
                (255, 0, 0),
                (bar_x, bar_y, bar_width, bar_height)
            )
            
            # Foreground (green)
            pygame.draw.rect(
                self.window,
                (0, 255, 0),
                (bar_x, bar_y, int(bar_width * hp_ratio), bar_height)
            )
            
            # Draw target line
            target_name = self.agent_states[agent]["target"]
            if target_name and not self.terminations.get(target_name, False):
                target_pos = self.agent_states[target_name]["pos"]
                target_center = (
                    target_pos[1] * self.cell_size + self.cell_size // 2,
                    target_pos[0] * self.cell_size + self.cell_size // 2
                )
                pygame.draw.line(
                    self.window,
                    target_line_color,
                    center,
                    target_center,
                    2
                )
        
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None