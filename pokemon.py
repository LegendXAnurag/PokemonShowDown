"""
Pokemon class for the RL training system.
"""

from pokemon_data import get_damage_multiplier


class Pokemon:
    """
    Represents a Pokemon in the game with its attributes and state.
    """
    
    def __init__(self, name, poke_type, sprite_path, base_health=100, base_attack=20):
        """
        Initialize a Pokemon.
        
        Args:
            name: Name of the Pokemon
            poke_type: Type of the Pokemon (Fire, Water, Grass, etc.)
            sprite_path: Path to the sprite image
            base_health: Base health points
            base_attack: Base attack damage
        """
        self.name = name
        self.poke_type = poke_type
        self.sprite_path = sprite_path
        self.base_health = base_health
        self.base_attack = base_attack
        
        # State variables
        self.health = base_health
        self.position = (0, 0)  # (x, y) position on grid
        self.is_alive = True
        self.team_id = None  # For team battles
        
    def reset(self):
        """Reset Pokemon to initial state."""
        self.health = self.base_health
        self.is_alive = True
        self.position = (0, 0)
        
    def take_damage(self, damage, attacker_type):
        """
        Apply damage to this Pokemon, considering type effectiveness.
        
        Args:
            damage: Base damage amount
            attacker_type: Type of the attacking Pokemon
            
        Returns:
            Actual damage dealt after type effectiveness
        """
        multiplier = get_damage_multiplier(attacker_type, self.poke_type)
        actual_damage = damage * multiplier
        
        self.health -= actual_damage
        if self.health <= 0:
            self.health = 0
            self.is_alive = False
            
        return actual_damage
        
    def attack_damage(self):
        """
        Get the base attack damage of this Pokemon.
        
        Returns:
            Base attack damage
        """
        return self.base_attack
        
    def set_position(self, x, y):
        """Set the position of the Pokemon on the grid."""
        self.position = (x, y)
        
    def set_team(self, team_id):
        """Set the team ID for team battles."""
        self.team_id = team_id
        
    def __repr__(self):
        return f"Pokemon({self.name}, {self.poke_type}, HP:{self.health}/{self.base_health}, Pos:{self.position})"
