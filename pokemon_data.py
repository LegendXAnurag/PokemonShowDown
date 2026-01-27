"""
Pokemon data including types, matchups, and sprite references
"""

# Type effectiveness chart (attacker -> defender -> multiplier)
TYPE_EFFECTIVENESS = {
    'fire': {
        'fire': 0.5,
        'water': 0.5,
        'grass': 2.0,
        'electric': 1.0,
        'ground': 1.0,
        'fighting': 1.0,
        'psychic': 1.0,
        'normal': 1.0
    },
    'water': {
        'fire': 2.0,
        'water': 0.5,
        'grass': 0.5,
        'electric': 1.0,
        'ground': 2.0,
        'fighting': 1.0,
        'psychic': 1.0,
        'normal': 1.0
    },
    'grass': {
        'fire': 0.5,
        'water': 2.0,
        'grass': 0.5,
        'electric': 1.0,
        'ground': 2.0,
        'fighting': 1.0,
        'psychic': 1.0,
        'normal': 1.0
    },
    'electric': {
        'fire': 1.0,
        'water': 2.0,
        'grass': 0.5,
        'electric': 0.5,
        'ground': 0.5,
        'fighting': 1.0,
        'psychic': 1.0,
        'normal': 1.0
    },
    'ground': {
        'fire': 2.0,
        'water': 1.0,
        'grass': 0.5,
        'electric': 2.0,
        'ground': 1.0,
        'fighting': 1.0,
        'psychic': 1.0,
        'normal': 1.0
    },
    'fighting': {
        'fire': 1.0,
        'water': 1.0,
        'grass': 1.0,
        'electric': 1.0,
        'ground': 1.0,
        'fighting': 1.0,
        'psychic': 0.5,
        'normal': 2.0
    },
    'psychic': {
        'fire': 1.0,
        'water': 1.0,
        'grass': 1.0,
        'electric': 1.0,
        'ground': 1.0,
        'fighting': 2.0,
        'psychic': 0.5,
        'normal': 1.0
    },
    'normal': {
        'fire': 1.0,
        'water': 1.0,
        'grass': 1.0,
        'electric': 1.0,
        'ground': 1.0,
        'fighting': 1.0,
        'psychic': 1.0,
        'normal': 1.0
    }
}

# Pokemon catalog with types
POKEMON_CATALOG = {
    'charizard': {
        'type': 'fire',
        'sprite': 'charizard.png',
        'color': (255, 100, 100)
    },
    'blastoise': {
        'type': 'water',
        'sprite': 'blastoise.png',
        'color': (100, 150, 255)
    },
    'venusaur': {
        'type': 'grass',
        'sprite': 'venusaur.png',
        'color': (100, 200, 100)
    },
    'pikachu': {
        'type': 'electric',
        'sprite': 'pikachu.png',
        'color': (255, 255, 100)
    },
    'sandslash': {
        'type': 'ground',
        'sprite': 'sandslash.png',
        'color': (180, 140, 100)
    },
    'machamp': {
        'type': 'fighting',
        'sprite': 'machamp.png',
        'color': (200, 100, 150)
    },
    'alakazam': {
        'type': 'psychic',
        'sprite': 'alakazam.png',
        'color': (255, 150, 255)
    },
    'snorlax': {
        'type': 'normal',
        'sprite': 'snorlax.png',
        'color': (150, 150, 150)
    }
}

# Type to index mapping for one-hot encoding
TYPE_TO_INDEX = {
    'fire': 0,
    'water': 1,
    'grass': 2,
    'electric': 3,
    'ground': 4,
    'fighting': 5,
    'psychic': 6,
    'normal': 7
}

INDEX_TO_TYPE = {v: k for k, v in TYPE_TO_INDEX.items()}

NUM_TYPES = len(TYPE_TO_INDEX)

def get_damage_multiplier(attacker_type, defender_type):
    """Get damage multiplier based on type matchup"""
    return TYPE_EFFECTIVENESS[attacker_type][defender_type]

def get_random_pokemon_names(n):
    """Get n random pokemon names from catalog"""
    import random
    pokemon_names = list(POKEMON_CATALOG.keys())
    return random.sample(pokemon_names, min(n, len(pokemon_names)))