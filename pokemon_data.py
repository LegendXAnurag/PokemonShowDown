# pokemon_data.py

POKEMON_DB = {
    "pikachu": {
        "name": "Pikachu",
        "hp": 100,
        "attack_power": 7,
        "model_path": "assets/pikachu.glb", 
        "color_fallback": (1.0, 1.0, 0.0), # Yellow
        "rotation_correction": 0 
    },
    "charmander": {
        "name": "Charmander",
        "hp": 100,
        "attack_power": 7,
        "model_path": "assets/charmander.glb",
        "color_fallback": (1.0, 0.0, 0.0), # Red
        "rotation_correction": 180 
    },
    "squirtle": {
        "name": "Squirtle",
        "hp": 100,
        "attack_power": 7,
        "model_path": "assets/squirtle.glb",
        "color_fallback": (0.0, 0.5, 1.0), # Blue
        "rotation_correction": 0
    },

    "rhyhorn": {
        "name": "Rhyhorn",
        "hp": 100,
        "attack_power": 7,
        "model_path": "assets/rhyhorn.glb",
        "color_fallback": (0.5, 0.5, 0.5),
        "rotation_correction": 0
    },
    "eevee": {
        "name": "Eevee",
        "hp": 100,
        "attack_power": 7,
        "model_path": "assets/eevee.glb",
        "color_fallback": (0.6, 0.4, 0.2), # Brown
        "rotation_correction": 0
    }
}