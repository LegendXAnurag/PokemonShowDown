# config.py
# Window Settings
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
WINDOW_TITLE = "Pokemon 3D Battle Engine"

# Simulation Settings
FPS = 60
DT = 1.0 / FPS         
GRID_SIZE = 10 
TILE_SIZE = 1.0
BOUNDARY = (GRID_SIZE * TILE_SIZE) / 2.0  # From -5 to 5
MAX_STEPS_PER_EPISODE = 4000 

# Pokemon Settings
NUM_AGENTS = 3  
POKEMON_SCALE_SIZE = 1.0 
HITBOX_RADIUS = 0.5    

# Vision / Lidar Settings
NUM_RAYS = 16
VISION_RANGE = 15.0  
# [CHANGE] Channels: Dist, IsWall, IsEnemy, EnemyHP, EnemyFacing
LIDAR_CHANNELS = 5   

# Attack Settings
ATTACK_RANGE = 2.0     
ATTACK_WIDTH = 1.0       
ATTACK_DURATION = 1.0  

# Gameplay
MOVE_SPEED = 0.10
ROTATION_SPEED = 4.0 

# --- RL / REWARD SETTINGS ---
REWARD_WIN = 100.0
REWARD_LOSS = -100.0
REWARD_DMG_DEALT_SCALE = 2.0   
REWARD_DMG_TAKEN_SCALE = 1.0   
REWARD_STEP_PENALTY = +0.03

# [CHANGE] Survival & Tactics Rewards
REWARD_BACKSTAB_BONUS = 2.0    # Multiplier for damage dealt from behind
REWARD_CRITICAL_SCALE = 4.0    # How much harder damage hurts when HP is low (1x at Full HP -> 4x at 0 HP)
REWARD_EXECUTE_SCALE = 2.0   # Greed: Multiplier for damage dealt to LOW HP opponents (Predatory behavior)

# --- SPAWN SETTINGS ---
SPAWN_MARGIN = 1.0       
MIN_SPAWN_DIST = 2.0     

# --- TRAINING HYPERPARAMETERS ---
LEARNING_RATE = 3e-4
TOTAL_TIMESTEPS = 5000000
NUM_STEPS = 256           
BATCH_SIZE = 512          
MINIBATCH_SIZE = 64
GAMMA = 0.99              
GAE_LAMBDA = 0.95         
CLIP_COEF = 0.2           
ENT_COEF = 0.01           
VF_COEF = 0.5             
MAX_GRAD_NORM = 0.5       
UPDATE_EPOCHS = 10        

# --- CHECKPOINTING ---
CHECKPOINT_FREQ = 2500   
CHECKPOINT_DIR = "checkpoints"
# MODEL_NAME = "pokemon_mappo_survival_no_backstab.pt"
MODEL_NAME = "no_bs_no_crit_v2_exec_backstab.pt"