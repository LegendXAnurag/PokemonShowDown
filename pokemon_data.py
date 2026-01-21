"""
Static data for Pokemon types, attributes, and sprite references.
This file contains the Pokemon data used in the RL training system.
"""

# Type effectiveness matrix
# Rows = attacker type, Columns = defender type
# Values: 2.0 = super effective, 1.0 = normal, 0.5 = not very effective
TYPE_EFFECTIVENESS = {
    'Fire': {'Fire': 0.5, 'Water': 0.5, 'Grass': 2.0, 'Electric': 1.0, 'Ice': 2.0, 'Fighting': 1.0, 'Psychic': 1.0, 'Dragon': 0.5},
    'Water': {'Fire': 2.0, 'Water': 0.5, 'Grass': 0.5, 'Electric': 1.0, 'Ice': 1.0, 'Fighting': 1.0, 'Psychic': 1.0, 'Dragon': 0.5},
    'Grass': {'Fire': 0.5, 'Water': 2.0, 'Grass': 0.5, 'Electric': 1.0, 'Ice': 1.0, 'Fighting': 1.0, 'Psychic': 1.0, 'Dragon': 0.5},
    'Electric': {'Fire': 1.0, 'Water': 2.0, 'Grass': 0.5, 'Electric': 0.5, 'Ice': 1.0, 'Fighting': 1.0, 'Psychic': 1.0, 'Dragon': 0.5},
    'Ice': {'Fire': 0.5, 'Water': 0.5, 'Grass': 2.0, 'Electric': 1.0, 'Ice': 0.5, 'Fighting': 1.0, 'Psychic': 1.0, 'Dragon': 2.0},
    'Fighting': {'Fire': 1.0, 'Water': 1.0, 'Grass': 1.0, 'Electric': 1.0, 'Ice': 2.0, 'Fighting': 1.0, 'Psychic': 0.5, 'Dragon': 1.0},
    'Psychic': {'Fire': 1.0, 'Water': 1.0, 'Grass': 1.0, 'Electric': 1.0, 'Ice': 1.0, 'Fighting': 2.0, 'Psychic': 0.5, 'Dragon': 1.0},
    'Dragon': {'Fire': 1.0, 'Water': 1.0, 'Grass': 1.0, 'Electric': 1.0, 'Ice': 1.0, 'Fighting': 1.0, 'Psychic': 1.0, 'Dragon': 2.0},
}

# Pokemon data: name, type, and sprite reference
POKEMON_LIST = [
    {
        'name': 'Charizard',
        'type': 'Fire',
        'sprite': 'sprites/charizard.png',
        'base_health': 100,
        'base_attack': 20
    },
    {
        'name': 'Blastoise',
        'type': 'Water',
        'sprite': 'sprites/blastoise.png',
        'base_health': 100,
        'base_attack': 20
    },
    {
        'name': 'Venusaur',
        'type': 'Grass',
        'sprite': 'sprites/venusaur.png',
        'base_health': 100,
        'base_attack': 20
    },
    {
        'name': 'Pikachu',
        'type': 'Electric',
        'sprite': 'sprites/pikachu.png',
        'base_health': 100,
        'base_attack': 20
    },
    {
        'name': 'Articuno',
        'type': 'Ice',
        'sprite': 'sprites/articuno.png',
        'base_health': 100,
        'base_attack': 20
    },
    {
        'name': 'Machamp',
        'type': 'Fighting',
        'sprite': 'sprites/machamp.png',
        'base_health': 100,
        'base_attack': 20
    },
    {
        'name': 'Mewtwo',
        'type': 'Psychic',
        'sprite': 'sprites/mewtwo.png',
        'base_health': 100,
        'base_attack': 20
    },
    {
        'name': 'Dragonite',
        'type': 'Dragon',
        'sprite': 'sprites/dragonite.png',
        'base_health': 100,
        'base_attack': 20
    },
]

def get_damage_multiplier(attacker_type, defender_type):
    """
    Get the damage multiplier based on type effectiveness.
    
    Args:
        attacker_type: Type of the attacking Pokemon
        defender_type: Type of the defending Pokemon
        
    Returns:
        Damage multiplier (2.0 = super effective, 1.0 = normal, 0.5 = not very effective)
    """
    return TYPE_EFFECTIVENESS.get(attacker_type, {}).get(defender_type, 1.0)
