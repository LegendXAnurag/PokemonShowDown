# config.py

# Window Settings
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 800
WINDOW_TITLE = "Pokemon 3D Battle Engine"

# Environment Settings
GRID_SIZE = 10  # 10x10 units
TILE_SIZE = 1.0
BOUNDARY = (GRID_SIZE * TILE_SIZE) / 2.0  # From -5 to 5

# Pokemon Settings
POKEMON_SCALE_SIZE = 1.0 
# Attack Settings
ATTACK_RANGE = 3.0       # 3x Pokemon size
ATTACK_WIDTH = 1.0       # 1x Pokemon size
ATTACK_DURATION = 10     # Frames the beam remains visible (approx 0.16s at 60fps)

# Camera Settings
# Views: (EyeX, EyeY, EyeZ, CenterX, CenterY, CenterZ)
VIEWS = [
    (0, 10, 10, 0, 0, 0),    # Front
    (10, 10, 0, 0, 0, 0),    # Right
    (0, 10, -10, 0, 0, 0),   # Back
    (-10, 10, 0, 0, 0, 0)    # Left
]

# Gameplay
MOVE_SPEED = 0.1
ROTATION_SPEED = 3.0 # Degrees