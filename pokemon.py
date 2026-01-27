"""
Pokemon class representing individual pokemon agents
"""
import numpy as np
from pokemon_data import POKEMON_CATALOG, TYPE_TO_INDEX, get_damage_multiplier
from config import BASE_HEALTH, BASE_ATTACK, POKEMON_SIZE, ATTACK_COOLDOWN

class Pokemon:
    def __init__(self, pokemon_name, agent_id, position, team_id=None):
        """
        Initialize a Pokemon
        
        Args:
            pokemon_name: Name of pokemon from catalog
            agent_id: Unique identifier for this agent
            position: Tuple (x, y) starting position
            team_id: Optional team identifier for team battles
        """
        if pokemon_name not in POKEMON_CATALOG:
            raise ValueError(f"Unknown pokemon: {pokemon_name}")
        
        pokemon_data = POKEMON_CATALOG[pokemon_name]
        
        self.name = pokemon_name
        self.agent_id = agent_id
        self.type = pokemon_data['type']
        self.type_index = TYPE_TO_INDEX[self.type]
        self.color = pokemon_data['color']
        self.sprite = pokemon_data['sprite']
        
        # Stats
        self.max_health = BASE_HEALTH
        self.health = BASE_HEALTH
        self.base_attack = BASE_ATTACK
        self.size = POKEMON_SIZE
        
        # Position
        self.x, self.y = position
        
        # Team (for team battles)
        self.team_id = team_id
        
        # Status
        self.alive = True
        self.last_damage_dealt = 0
        self.last_damage_taken = 0
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        self.kills = 0
        
        # Attack cooldown
        self.attack_cooldown = 0  # Starts at 0 (can attack immediately)
        
    def reset(self, position):
        """Reset pokemon to initial state at new position"""
        self.health = self.max_health
        self.x, self.y = position
        self.alive = True
        self.last_damage_dealt = 0
        self.last_damage_taken = 0
        self.total_damage_dealt = 0
        self.total_damage_taken = 0
        self.kills = 0
        self.attack_cooldown = 0
        
    def move(self, dx, dy, grid_size, occupied_positions):
        """
        Move pokemon by (dx, dy) if valid
        
        Args:
            dx, dy: Movement direction
            grid_size: Size of the grid
            occupied_positions: Set of (x, y) positions that are occupied (obstacles + other pokemon)
            
        Returns:
            bool: True if move was successful
        """
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Check boundaries
        if new_x < 0 or new_x >= grid_size or new_y < 0 or new_y >= grid_size:
            return False
        
        # Check if position is occupied (by obstacle or another pokemon)
        if (new_x, new_y) in occupied_positions:
            return False
        
        self.x = new_x
        self.y = new_y
        return True
    
    def update_cooldown(self):
        """Decrease attack cooldown by 1 (called each step)"""
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1
    
    def can_attack(self):
        """Check if pokemon can attack (cooldown is 0)"""
        return self.attack_cooldown == 0
    
    def attack(self, direction, targets, obstacles, grid_size):
        """
        Attack in a direction (beam attack)
        
        Args:
            direction: Tuple (dx, dy) attack direction
            targets: List of other Pokemon objects
            obstacles: Set of (x, y) obstacle positions
            grid_size: Size of the grid
            
        Returns:
            List of (target, damage) tuples for hit targets, or None if on cooldown
        """
        # Check cooldown
        if not self.can_attack():
            return None  # Can't attack yet
        
        dx, dy = direction
        hits = []
        
        # Trace beam from current position
        current_x, current_y = self.x, self.y
        
        while True:
            current_x += dx
            current_y += dy
            
            # Check boundaries
            if current_x < 0 or current_x >= grid_size or current_y < 0 or current_y >= grid_size:
                break
            
            # Check if obstacle blocks beam
            if (current_x, current_y) in obstacles:
                break
            
            # Check if any target is hit
            for target in targets:
                if not target.alive or target.agent_id == self.agent_id:
                    continue
                    
                if target.x == current_x and target.y == current_y:
                    # Calculate damage with type effectiveness
                    multiplier = get_damage_multiplier(self.type, target.type)
                    damage = self.base_attack * multiplier
                    
                    # Apply damage
                    target.take_damage(damage)
                    self.last_damage_dealt = damage
                    self.total_damage_dealt += damage
                    
                    if not target.alive:
                        self.kills += 1
                    
                    hits.append((target, damage))
                    # Beam stops after hitting a target
                    # Set cooldown after successful attack
                    self.attack_cooldown = ATTACK_COOLDOWN
                    return hits
        
        # Attack missed - still set cooldown
        self.last_damage_dealt = 0
        self.attack_cooldown = ATTACK_COOLDOWN
        return hits
    
    def take_damage(self, damage):
        """
        Take damage
        
        Args:
            damage: Amount of damage to take
        """
        self.health -= damage
        self.last_damage_taken = damage
        self.total_damage_taken += damage
        
        if self.health <= 0:
            self.health = 0
            self.alive = False
    
    def get_health_ratio(self):
        """Get health as ratio of max health"""
        return self.health / self.max_health
    
    def get_position(self):
        """Get current position"""
        return (self.x, self.y)
    
    def __repr__(self):
        return f"Pokemon({self.name}, id={self.agent_id}, type={self.type}, hp={self.health}/{self.max_health}, pos=({self.x},{self.y}))"