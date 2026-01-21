"""
pokemon.py - Pokemon class definition for Pokemon RL environment
"""

import numpy as np
from typing import List, Tuple, Optional


class PokemonType:
    """Pokemon type definitions and type chart"""
    
    # Available types
    FIRE = "fire"
    WATER = "water"
    GRASS = "grass"
    ELECTRIC = "electric"
    GROUND = "ground"
    FLYING = "flying"
    PSYCHIC = "psychic"
    FIGHTING = "fighting"
    ROCK = "rock"
    
    # Type chart: (attacker_type, defender_type) -> multiplier
    TYPE_CHART = {
        # Classic trio
        (FIRE, GRASS): 2.0,
        (GRASS, WATER): 2.0,
        (WATER, FIRE): 2.0,
        (GRASS, FIRE): 0.5,
        (WATER, GRASS): 0.5,
        (FIRE, WATER): 0.5,
        
        # Electric advantages
        (ELECTRIC, WATER): 2.0,
        (ELECTRIC, FLYING): 2.0,
        (ELECTRIC, GROUND): 0.0,  # No effect
        (GROUND, ELECTRIC): 2.0,
        
        # Ground advantages
        (GROUND, FIRE): 2.0,
        (GROUND, ELECTRIC): 2.0,
        (GROUND, ROCK): 2.0,
        (FLYING, GROUND): 0.0,
        
        # Flying advantages
        (FLYING, GRASS): 2.0,
        (FLYING, FIGHTING): 2.0,
        (ROCK, FLYING): 2.0,
        (ELECTRIC, FLYING): 2.0,
        
        # Fighting advantages
        (FIGHTING, ROCK): 2.0,
        (PSYCHIC, FIGHTING): 2.0,
        (FLYING, FIGHTING): 2.0,
        
        # Psychic advantages
        (PSYCHIC, FIGHTING): 2.0,
        
        # Rock advantages
        (ROCK, FIRE): 2.0,
        (ROCK, FLYING): 2.0,
        (WATER, ROCK): 2.0,
        (GRASS, ROCK): 2.0,
        (FIGHTING, ROCK): 2.0,
        (GROUND, ROCK): 2.0,
    }
    
    @staticmethod
    def get_multiplier(attacker_type: str, defender_type: str) -> float:
        """Get type advantage multiplier"""
        return PokemonType.TYPE_CHART.get((attacker_type, defender_type), 1.0)
    
    @staticmethod
    def get_all_types() -> List[str]:
        """Get list of all available types"""
        return [
            PokemonType.FIRE,
            PokemonType.WATER,
            PokemonType.GRASS,
            PokemonType.ELECTRIC,
            PokemonType.GROUND,
            PokemonType.FLYING,
            PokemonType.PSYCHIC,
            PokemonType.FIGHTING,
            PokemonType.ROCK,
        ]


class Pokemon:
    """Pokemon class with stats and properties"""
    
    def __init__(
        self,
        name: str,
        poke_type: str,
        max_hp: float = 100.0,
        base_attack: float = 10.0,
        base_defense: float = 0.0,
        speed: int = 1,
        color: Tuple[int, int, int] = (128, 128, 128),
    ):
        """
        Initialize a Pokemon
        
        Args:
            name: Pokemon name (e.g., "Charmander")
            poke_type: Pokemon type (use PokemonType constants)
            max_hp: Maximum HP
            base_attack: Base attack damage
            base_defense: Damage reduction (0.0 = no reduction, 0.5 = 50% reduction)
            speed: Movement speed (currently not used, for future expansion)
            color: RGB color tuple for rendering
        """
        self.name = name
        self.poke_type = poke_type
        self.max_hp = max_hp
        self.base_attack = base_attack
        self.base_defense = base_defense
        self.speed = speed
        self.color = color
        
        # Validate type
        if poke_type not in PokemonType.get_all_types():
            raise ValueError(f"Invalid pokemon type: {poke_type}")
    
    def calculate_damage(self, defender: 'Pokemon') -> float:
        """
        Calculate damage dealt to defender
        
        Args:
            defender: Pokemon being attacked
            
        Returns:
            Damage amount
        """
        # Get type advantage multiplier
        multiplier = PokemonType.get_multiplier(self.poke_type, defender.poke_type)
        
        # Calculate raw damage
        raw_damage = self.base_attack * multiplier
        
        # Apply defender's defense
        damage = raw_damage * (1.0 - defender.base_defense)
        
        return max(0.0, damage)
    
    def get_weaknesses(self) -> List[str]:
        """Get list of types this Pokemon is weak against"""
        weaknesses = []
        for attacker_type in PokemonType.get_all_types():
            multiplier = PokemonType.get_multiplier(attacker_type, self.poke_type)
            if multiplier > 1.0:
                weaknesses.append(attacker_type)
        return weaknesses
    
    def get_resistances(self) -> List[str]:
        """Get list of types this Pokemon resists"""
        resistances = []
        for attacker_type in PokemonType.get_all_types():
            multiplier = PokemonType.get_multiplier(attacker_type, self.poke_type)
            if 0.0 < multiplier < 1.0:
                resistances.append(attacker_type)
        return resistances
    
    def get_immunities(self) -> List[str]:
        """Get list of types this Pokemon is immune to"""
        immunities = []
        for attacker_type in PokemonType.get_all_types():
            multiplier = PokemonType.get_multiplier(attacker_type, self.poke_type)
            if multiplier == 0.0:
                immunities.append(attacker_type)
        return immunities
    
    def get_advantages(self) -> List[str]:
        """Get list of types this Pokemon is strong against"""
        advantages = []
        for defender_type in PokemonType.get_all_types():
            multiplier = PokemonType.get_multiplier(self.poke_type, defender_type)
            if multiplier > 1.0:
                advantages.append(defender_type)
        return advantages
    
    def __repr__(self):
        return f"Pokemon(name='{self.name}', type='{self.poke_type}', hp={self.max_hp}, atk={self.base_attack})"


# ============================================================================
# Predefined Pokemon
# ============================================================================

def create_charmander() -> Pokemon:
    """Create a Charmander (Fire type)"""
    return Pokemon(
        name="Charmander",
        poke_type=PokemonType.FIRE,
        max_hp=100.0,
        base_attack=12.0,
        base_defense=0.0,
        speed=1,
        color=(255, 69, 0),  # Red-Orange
    )


def create_bulbasaur() -> Pokemon:
    """Create a Bulbasaur (Grass type)"""
    return Pokemon(
        name="Bulbasaur",
        poke_type=PokemonType.GRASS,
        max_hp=110.0,
        base_attack=10.0,
        base_defense=0.1,  # 10% damage reduction
        speed=1,
        color=(34, 139, 34),  # Forest Green
    )


def create_squirtle() -> Pokemon:
    """Create a Squirtle (Water type)"""
    return Pokemon(
        name="Squirtle",
        poke_type=PokemonType.WATER,
        max_hp=105.0,
        base_attack=11.0,
        base_defense=0.05,  # 5% damage reduction
        speed=1,
        color=(30, 144, 255),  # Dodger Blue
    )


def create_pikachu() -> Pokemon:
    """Create a Pikachu (Electric type)"""
    return Pokemon(
        name="Pikachu",
        poke_type=PokemonType.ELECTRIC,
        max_hp=90.0,
        base_attack=13.0,
        base_defense=0.0,
        speed=2,  # Faster
        color=(255, 215, 0),  # Gold
    )


def create_geodude() -> Pokemon:
    """Create a Geodude (Rock/Ground type - using Ground)"""
    return Pokemon(
        name="Geodude",
        poke_type=PokemonType.GROUND,
        max_hp=120.0,
        base_attack=9.0,
        base_defense=0.2,  # 20% damage reduction (tanky)
        speed=1,
        color=(139, 90, 43),  # Saddle Brown
    )


def create_pidgey() -> Pokemon:
    """Create a Pidgey (Flying type)"""
    return Pokemon(
        name="Pidgey",
        poke_type=PokemonType.FLYING,
        max_hp=85.0,
        base_attack=11.0,
        base_defense=0.0,
        speed=2,
        color=(169, 169, 169),  # Dark Gray
    )


def create_machop() -> Pokemon:
    """Create a Machop (Fighting type)"""
    return Pokemon(
        name="Machop",
        poke_type=PokemonType.FIGHTING,
        max_hp=95.0,
        base_attack=14.0,  # High attack
        base_defense=0.0,
        speed=1,
        color=(192, 192, 192),  # Silver
    )


def create_abra() -> Pokemon:
    """Create an Abra (Psychic type)"""
    return Pokemon(
        name="Abra",
        poke_type=PokemonType.PSYCHIC,
        max_hp=80.0,
        base_attack=12.0,
        base_defense=0.0,
        speed=2,
        color=(255, 105, 180),  # Hot Pink
    )


def create_onix() -> Pokemon:
    """Create an Onix (Rock type)"""
    return Pokemon(
        name="Onix",
        poke_type=PokemonType.ROCK,
        max_hp=130.0,
        base_attack=8.0,
        base_defense=0.25,  # Very tanky
        speed=1,
        color=(105, 105, 105),  # Dim Gray
    )


# ============================================================================
# Pokemon Registry
# ============================================================================

POKEMON_REGISTRY = {
    "charmander": create_charmander,
    "bulbasaur": create_bulbasaur,
    "squirtle": create_squirtle,
    "pikachu": create_pikachu,
    "geodude": create_geodude,
    "pidgey": create_pidgey,
    "machop": create_machop,
    "abra": create_abra,
    "onix": create_onix,
}


def get_pokemon(name: str) -> Pokemon:
    """
    Get a Pokemon by name
    
    Args:
        name: Pokemon name (case-insensitive)
        
    Returns:
        Pokemon instance
        
    Raises:
        ValueError: If Pokemon name not found
    """
    name_lower = name.lower()
    if name_lower not in POKEMON_REGISTRY:
        available = ", ".join(POKEMON_REGISTRY.keys())
        raise ValueError(f"Pokemon '{name}' not found. Available: {available}")
    
    return POKEMON_REGISTRY[name_lower]()


def get_available_pokemon() -> List[str]:
    """Get list of all available Pokemon names"""
    return list(POKEMON_REGISTRY.keys())


def create_balanced_team(team_size: int = 3) -> List[Pokemon]:
    """
    Create a balanced team of Pokemon with different types
    
    Args:
        team_size: Number of Pokemon in team
        
    Returns:
        List of Pokemon instances
    """
    if team_size > len(POKEMON_REGISTRY):
        raise ValueError(f"Cannot create team of size {team_size}, only {len(POKEMON_REGISTRY)} Pokemon available")
    
    # Default balanced teams
    if team_size == 3:
        return [create_charmander(), create_bulbasaur(), create_squirtle()]
    elif team_size == 4:
        return [create_charmander(), create_bulbasaur(), create_squirtle(), create_pikachu()]
    elif team_size == 6:
        return [
            create_charmander(),
            create_bulbasaur(), 
            create_squirtle(),
            create_pikachu(),
            create_geodude(),
            create_pidgey(),
        ]
    else:
        # Random selection
        import random
        pokemon_names = random.sample(list(POKEMON_REGISTRY.keys()), team_size)
        return [get_pokemon(name) for name in pokemon_names]