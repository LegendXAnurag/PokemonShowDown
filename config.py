"""
Configuration file for Pokemon Showdown RL Training
"""
# Environment Configuration
ENV_CONFIG = {
    'level_1': {
        'grid_size': 5,
        'num_obstacles': 0,
        'number_agents': 2,
        'max_steps': 200
    },
    'level_2': {
        'grid_size': 6,
        'num_obstacles': 0,
        'number_agents': 2,
        'max_steps': 250
    },
    'level_3': {
        'grid_size': 6,
        'num_obstacles': 3,
        'number_agents': 2,
        'max_steps': 300
    },
    'level_4': {
        'grid_size': 10,
        'num_obstacles': 5,
        'number_agents': 3,
        'max_steps': 400
    },
    'level_5': {
        'grid_size': 10,
        'num_obstacles': 5,
        'number_agents': 5,
        'max_steps': 500
    },
    'level_6': {
        'grid_size': 10,
        'num_obstacles': 5,
        'number_agents': 4,
        'teams': [[0, 1], [2, 3]],  # 2v2
        'max_steps': 500
    }
}

# Pokemon Base Stats
BASE_HEALTH = 100
BASE_ATTACK = 10
POKEMON_SIZE = 1

# Action Space
NUM_ACTIONS = 8
ACTIONS = {
    0: 'move_up',
    1: 'move_down',
    2: 'move_left',
    3: 'move_right',
    4: 'attack_up',
    5: 'attack_down',
    6: 'attack_left',
    7: 'attack_right'
}

# Reward Configuration
REWARDS = {
    'damage_dealt': 0.5,
    'damage_taken': -0.5,
    'opponent_defeated': 100,
    'getting_defeated': -100,
    'victory': 200,
    'time_penalty': -1,
    'team_damage': -50,
    'missed_attack': -2.0  # Penalty for attacks that don't hit
}

# Attack cooldown (number of steps before can attack again)
ATTACK_COOLDOWN = 3

# Training Hyperparameters
TRAINING_CONFIG = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'entropy_coef_start': 0.15,
    'entropy_coef_end': 0.01,
    'entropy_decay_steps': 100000,
    'value_loss_coef': 0.5,
    'max_grad_norm': 0.5,
    'num_epochs': 10,
    'batch_size': 32768,  # Increased from 8192 to 32768 for faster training
    'num_minibatches': 64,  # Increased from 32 to 64 to keep minibatch size reasonable
    'total_timesteps': 1000000,
    'save_interval': 100000,
    'log_interval': 10000  # Increased from 1000 to log less frequently
}

# Network Architecture
NETWORK_CONFIG = {
    'hidden_dim': 256,
    'num_layers': 3,
    'activation': 'relu'
}

# Visualization
RENDER_CONFIG = {
    'cell_size': 100,
    'fps': 45,  # FPS for visualization mode (high for smooth beam animation)
    'training_fps': 1000,  # Very high FPS for training (fast simulation)
    'beam_speed': 0.7,  # Beam speed in cells per frame (at visualization FPS)
    'colors': {
        'background': (240, 240, 240),
        'grid': (200, 200, 200),
        'obstacle': (100, 100, 100),
        'fire': (255, 100, 100),
        'water': (100, 150, 255),
        'grass': (100, 200, 100),
        'electric': (255, 255, 100),
        'ground': (180, 140, 100),
        'fighting': (200, 100, 150),
        'psychic': (255, 150, 255),
        'normal': (150, 150, 150)
    }
}