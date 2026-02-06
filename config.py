# config.py
# Window Settings
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
WINDOW_TITLE = "Pokemon 3D Team Battle Engine"

# Simulation Settings
FPS = 60
DT = 1.0 / FPS         
GRID_SIZE = 10 
TILE_SIZE = 1.0
BOUNDARY = (GRID_SIZE * TILE_SIZE) / 2.0  # From -5 to 5
MAX_STEPS_PER_EPISODE = 4000 

# --- TEAM SETTINGS ---
# Define teams: Count of agents per team, and their Color (R, G, B)
TEAMS_SETUP = [
    {"count": 2, "color": (0.85, 0.1, 0.1)},  # Team 0: Red
    {"count": 2, "color": (0.1, 0.1, 0.85)},  # Team 1: Blue
    # {"count": 1, "color": (0.1, 0.8, 0.1)}, # Team 2: Green (Example of uneven teams)
]

# Calculate total agents dynamically
NUM_AGENTS = sum(t["count"] for t in TEAMS_SETUP)

POKEMON_SCALE_SIZE = 1.0 
HITBOX_RADIUS = 0.5    

ALLOW_BACKWARD=False

# Vision / Lidar Settings
NUM_RAYS = 16
VISION_RANGE = 15.0  
# [CHANGE] Channels: Dist, IsWall, IsEnemy, IsTeammate, UnitHP, UnitFacing
LIDAR_CHANNELS = 6   

# Attack Settings
ATTACK_RANGE = 2.0     
ATTACK_WIDTH = 1.0       
ATTACK_DURATION = 1.0  

# Gameplay
MOVE_SPEED = 0.10
ROTATION_SPEED = 4.0 

# --- RL / REWARD SETTINGS ---
REWARD_WIN = 200.0
REWARD_LOSS = -100.0
DEATH_PENALTY = -50.0
TIMEOUT_LOSS = -1000.0
REWARD_DMG_DEALT_SCALE = 2.5   
REWARD_DMG_TAKEN_SCALE = 1.0   
REWARD_STEP_PENALTY = +0.03

# Survival & Tactics Rewards
REWARD_BACKSTAB_BONUS = 1.1    
REWARD_CRITICAL_SCALE = 1.3    
REWARD_EXECUTE_SCALE = 1.8   
REWARD_FRIENDLY_FIRE_SCALE = 5000.0 # [NEW] Penalty multiplier for hitting teammates

# --- SPAWN SETTINGS ---
SPAWN_MARGIN = 1.0       
MIN_SPAWN_DIST = 2.0     

# --- TRAINING HYPERPARAMETERS ---
LEARNING_RATE = 3e-4
TOTAL_TIMESTEPS = 9000000
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
MODEL_NAME = "pokemon_team_battle_2v2_backstab_feb6.pt"