# pokemon_data.py

POKEMON_DB = {
    "pikachu": {
        "name": "Pikachu",
        "hp": 100,
        "attack_power": 25,
        "model_path": "assets/pikachu.glb", 
        "color_fallback": (1.0, 1.0, 0.0), # Yellow
        "rotation_correction": 0 # Facing correct way by default
    },
    "charmander": {
        "name": "Charmander",
        "hp": 100,
        "attack_power": 25,
        "model_path": "assets/charmander.glb",
        "color_fallback": (1.0, 0.0, 0.0), # Red
        # Fixes the model facing backwards. 
        # Rotates mesh 180 degrees around Y-axis during load.
        "rotation_correction": 180 
    }
}