"""
PettingZoo Parallel Environment for Pokemon Showdown
"""
import numpy as np
import functools
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pokemon import Pokemon
from pokemon_data import get_random_pokemon_names, NUM_TYPES
from config import ENV_CONFIG, NUM_ACTIONS, ACTIONS, REWARDS, ATTACK_COOLDOWN

class PokemonShowdownEnv(ParallelEnv):
    metadata = {
        "name": "pokemon_showdown_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
    }
    
    def __init__(self, level='level_1', render_mode=None):
        """
        Initialize Pokemon Showdown Environment
        
        Args:
            level: Training level (level_1 to level_6)
            render_mode: Rendering mode
        """
        super().__init__()
        
        self.level = level
        self.config = ENV_CONFIG[level]
        self.render_mode = render_mode
        
        self.grid_size = self.config['grid_size']
        self.num_obstacles = self.config['num_obstacles']
        self.number_agents = self.config['number_agents']
        self.max_steps = self.config['max_steps']
        self.teams = self.config.get('teams', None)
        
        # Generate agent names
        self.possible_agents = [f"agent_{i}" for i in range(self.number_agents)]
        
        # Initialize spaces
        self._setup_spaces()
        
        # State
        self.agents = []
        self.pokemons = {}
        self.obstacles = set()
        self.current_step = 0
        self.episode_rewards = {agent: 0 for agent in self.possible_agents}
        
        # Pending attacks for beam visualization
        self.pending_attacks = []
        
        # Renderer
        self.renderer = None
        
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # Observation space:
        # - Grid: grid_size x grid_size x (number_agents + 1 for obstacles)
        # - Own health: 1
        # - Own type: NUM_TYPES (one-hot)
        # - Own cooldown: 1 (normalized)
        # - Other agents info: (number_agents - 1) x (2 pos + 1 health + NUM_TYPES type + 1 cooldown)
        
        grid_channels = self.number_agents + 1  # agents + obstacles
        own_info_size = 1 + NUM_TYPES + 1  # health + type + cooldown
        other_info_size = (self.number_agents - 1) * (2 + 1 + NUM_TYPES + 1)  # pos + health + type + cooldown per agent
        
        obs_size = (self.grid_size * self.grid_size * grid_channels) + own_info_size + other_info_size
        
        self.observation_spaces = {
            agent: spaces.Box(
                low=0, high=1, shape=(obs_size,), dtype=np.float32
            )
            for agent in self.possible_agents
        }
        
        self.action_spaces = {
            agent: spaces.Discrete(NUM_ACTIONS)
            for agent in self.possible_agents
        }
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
        
        self.agents = self.possible_agents[:]
        self.current_step = 0
        self.episode_rewards = {agent: 0 for agent in self.possible_agents}
        self.pending_attacks = []
        
        # Generate obstacles (ensuring they don't share edges/vertices)
        self.obstacles = self._generate_obstacles()
        
        # Generate random pokemon names and positions
        pokemon_names = get_random_pokemon_names(self.number_agents)
        positions = self._generate_start_positions()
        
        # Create pokemon instances
        self.pokemons = {}
        for i, agent in enumerate(self.possible_agents):
            team_id = None
            if self.teams:
                # Find which team this agent belongs to
                for team_idx, team in enumerate(self.teams):
                    if i in team:
                        team_id = team_idx
                        break
            
            pokemon = Pokemon(
                pokemon_names[i],
                agent_id=agent,
                position=positions[i],
                team_id=team_id
            )
            self.pokemons[agent] = pokemon
        
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def _generate_obstacles(self):
        """Generate obstacle positions that don't share edges or vertices"""
        obstacles = set()
        attempts = 0
        max_attempts = 1000
        
        while len(obstacles) < self.num_obstacles and attempts < max_attempts:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            
            # Check if position is valid (no adjacent obstacles)
            valid = True
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if (x + dx, y + dy) in obstacles:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                obstacles.add((x, y))
            
            attempts += 1
        
        return obstacles
    
    def _generate_start_positions(self):
        """Generate starting positions for all agents (no overlapping)"""
        positions = []
        occupied = set(self.obstacles)
        
        for _ in range(self.number_agents):
            while True:
                x = np.random.randint(0, self.grid_size)
                y = np.random.randint(0, self.grid_size)
                
                if (x, y) not in occupied:
                    positions.append((x, y))
                    occupied.add((x, y))
                    break
        
        return positions
    
    def _get_observation(self, agent):
        """Get observation for a specific agent"""
        pokemon = self.pokemons[agent]
        
        # Grid representation (flattened)
        grid = np.zeros((self.grid_size, self.grid_size, self.number_agents + 1), dtype=np.float32)
        
        # Mark obstacles
        for ox, oy in self.obstacles:
            grid[ox, oy, 0] = 1.0
        
        # Mark agents
        for i, other_agent in enumerate(self.possible_agents):
            other_pokemon = self.pokemons[other_agent]
            if other_pokemon.alive:
                grid[other_pokemon.x, other_pokemon.y, i + 1] = 1.0
        
        grid_flat = grid.flatten()
        
        # Own info
        own_health = np.array([pokemon.get_health_ratio()], dtype=np.float32)
        own_type = np.zeros(NUM_TYPES, dtype=np.float32)
        own_type[pokemon.type_index] = 1.0
        # Normalize cooldown (0 = ready, 1 = max cooldown)
        own_cooldown = np.array([pokemon.attack_cooldown / ATTACK_COOLDOWN], dtype=np.float32)
        
        # Other agents info
        other_info = []
        for other_agent in self.possible_agents:
            if other_agent == agent:
                continue
            
            other_pokemon = self.pokemons[other_agent]
            
            # Normalized position
            pos = np.array([
                other_pokemon.x / self.grid_size,
                other_pokemon.y / self.grid_size
            ], dtype=np.float32)
            
            # Health
            health = np.array([other_pokemon.get_health_ratio()], dtype=np.float32)
            
            # Type (one-hot)
            ptype = np.zeros(NUM_TYPES, dtype=np.float32)
            if other_pokemon.alive:
                ptype[other_pokemon.type_index] = 1.0
            
            # Cooldown (normalized)
            cooldown = np.array([other_pokemon.attack_cooldown / ATTACK_COOLDOWN], dtype=np.float32)
            
            other_info.extend([*pos, *health, *ptype, *cooldown])
        
        other_info = np.array(other_info, dtype=np.float32)
        
        # Concatenate all observations
        obs = np.concatenate([grid_flat, own_health, own_type, own_cooldown, other_info])
        
        return obs
    
    def _get_occupied_positions(self):
        """Get all currently occupied positions"""
        occupied = set(self.obstacles)
        for pokemon in self.pokemons.values():
            if pokemon.alive:
                occupied.add((pokemon.x, pokemon.y))
        return occupied
    
    def step(self, actions):
        """Execute one step of the environment"""
        self.current_step += 1
        
        # Execute actions
        rewards = {agent: 0.0 for agent in self.agents}
        
        # Reset last damage counters and update cooldowns
        for pokemon in self.pokemons.values():
            pokemon.last_damage_dealt = 0
            pokemon.last_damage_taken = 0
            pokemon.update_cooldown()  # Decrease cooldown each step
        
        # Clear pending attacks
        self.pending_attacks = []
        
        # Get occupied positions before movements
        occupied_before = self._get_occupied_positions()
        
        # Process all actions
        for agent, action in actions.items():
            if agent not in self.agents:
                continue
            
            pokemon = self.pokemons[agent]
            if not pokemon.alive:
                continue
            
            action_name = ACTIONS[action]
            
            # Movement actions
            if action_name.startswith('move_'):
                direction_map = {
                    'move_up': (0, -1),
                    'move_down': (0, 1),
                    'move_left': (-1, 0),
                    'move_right': (1, 0)
                }
                dx, dy = direction_map[action_name]
                
                # Remove current position from occupied before checking
                occupied_before.discard((pokemon.x, pokemon.y))
                
                # Attempt move with collision checking
                pokemon.move(dx, dy, self.grid_size, occupied_before)
                
                # Add new position back to occupied
                occupied_before.add((pokemon.x, pokemon.y))
            
            # Attack actions
            elif action_name.startswith('attack_'):
                direction_map = {
                    'attack_up': (0, -1),
                    'attack_down': (0, 1),
                    'attack_left': (-1, 0),
                    'attack_right': (1, 0)
                }
                direction = direction_map[action_name]
                
                # Get all valid targets
                targets = [p for a, p in self.pokemons.items() if a != agent and p.alive]
                
                # Try to attack (returns None if on cooldown)
                hits = pokemon.attack(direction, targets, self.obstacles, self.grid_size)
                
                if hits is None:
                    # Attack was blocked by cooldown - no penalty, no visualization
                    pass
                else:
                    # Attack was executed (even if it missed)
                    # Store attack for visualization
                    self.pending_attacks.append({
                        'attacker': agent,
                        'start_pos': (pokemon.x, pokemon.y),
                        'direction': direction,
                        'color': pokemon.color
                    })
                    
                    if len(hits) == 0:
                        # Attack missed - apply penalty
                        rewards[agent] += REWARDS['missed_attack']
                    else:
                        # Attack hit - calculate rewards
                        for target, damage in hits:
                            # Check if friendly fire (team battle)
                            is_friendly_fire = False
                            if self.teams and pokemon.team_id is not None:
                                is_friendly_fire = pokemon.team_id == target.team_id
                            
                            if is_friendly_fire:
                                rewards[agent] += REWARDS['team_damage']
                            else:
                                rewards[agent] += damage * REWARDS['damage_dealt']
                                
                                if not target.alive:
                                    rewards[agent] += REWARDS['opponent_defeated']
        
        # Apply damage penalties
        for agent in self.agents:
            pokemon = self.pokemons[agent]
            if pokemon.last_damage_taken > 0:
                rewards[agent] += pokemon.last_damage_taken * REWARDS['damage_taken']
            
            if not pokemon.alive:
                rewards[agent] += REWARDS['getting_defeated']
        
        # Time penalty
        for agent in self.agents:
            if self.pokemons[agent].alive:
                rewards[agent] += REWARDS['time_penalty']
        
        # Update alive agents
        self.agents = [agent for agent in self.agents if self.pokemons[agent].alive]
        
        # Check termination
        terminated = {}
        truncated = {}
        
        alive_teams = set()
        if self.teams:
            for agent in self.agents:
                team_id = self.pokemons[agent].team_id
                if team_id is not None:
                    alive_teams.add(team_id)
        
        # Episode ends if:
        # 1. Only one agent/team remains
        # 2. Max steps reached
        # 3. All agents dead
        
        if self.teams:
            episode_done = len(alive_teams) <= 1 or self.current_step >= self.max_steps
        else:
            episode_done = len(self.agents) <= 1 or self.current_step >= self.max_steps
        
        # Victory rewards
        if episode_done and len(self.agents) > 0:
            if self.teams:
                # Reward winning team
                winning_team = list(alive_teams)[0] if len(alive_teams) == 1 else None
                if winning_team is not None:
                    for agent in self.agents:
                        if self.pokemons[agent].team_id == winning_team:
                            rewards[agent] += REWARDS['victory']
            else:
                # Reward last survivor
                if len(self.agents) == 1:
                    rewards[self.agents[0]] += REWARDS['victory']
        
        for agent in self.possible_agents:
            terminated[agent] = episode_done
            truncated[agent] = False
        
        # Get observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        # Update episode rewards
        for agent, reward in rewards.items():
            self.episode_rewards[agent] += reward
        
        return observations, rewards, terminated, truncated, infos
    
    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return
        
        if self.renderer is None:
            from visualizer import PokemonRenderer
            self.renderer = PokemonRenderer(self.grid_size)
        
        # Add pending attacks as beams
        for attack in self.pending_attacks:
            self.renderer.add_beam(
                attack['start_pos'][0],
                attack['start_pos'][1],
                attack['direction'],
                attack['color'],
                attack['attacker']
            )
        
        return self.renderer.render(
            self.pokemons,
            self.obstacles,
            self.grid_size,
            mode=self.render_mode
        )
    
    def close(self):
        """Clean up resources"""
        if self.renderer is not None:
            self.renderer.close()