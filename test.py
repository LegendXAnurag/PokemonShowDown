import argparse
import os
import math
import numpy as np
import torch
import pygame
import random
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Local Imports
import config
import pokemon_data
from ground import draw_ground, draw_walls
from pokemon import Pokemon
from model import PokemonAgent

def get_random_spawn(existing_positions=[]):
    """
    Generates a random (x, z) position and angle.
    Ensures the position is within bounds and maintains MIN_SPAWN_DIST 
    from any existing positions provided.
    """
    limit = config.BOUNDARY - config.SPAWN_MARGIN
    
    while True:
        x = random.uniform(-limit, limit)
        z = random.uniform(-limit, limit)
        angle = random.uniform(0, 360)
        
        valid = True
        for (ex, ez) in existing_positions:
            dist = math.sqrt((x - ex)**2 + (z - ez)**2)
            if dist < config.MIN_SPAWN_DIST:
                valid = False
                break
        
        if valid:
            return x, z, angle

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Trained Pokemon Agents")
    parser.add_argument("--model-path", type=str, default=f"{config.CHECKPOINT_DIR}/{config.MODEL_NAME}",
        help="path to the trained model checkpoint")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    return parser.parse_args()

def init_gl():
    # Day Blue Sky
    glClearColor(0.53, 0.81, 0.92, 1.0) 
    
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (10, 15, 10, 0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1.0))
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
    
    # Transparency for beam/walls
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (config.SCREEN_WIDTH / config.SCREEN_HEIGHT), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

def draw_hp_bar(pokemon, camera_pos):
    """ Billboarding HP Bar """
    glPushMatrix()
    glTranslatef(pokemon.x, pokemon.y + 2.0, pokemon.z)
    
    # Billboarding: Calculate angle to face camera
    cam_x, cam_y, cam_z = camera_pos[0], camera_pos[1], camera_pos[2]
    dx = cam_x - pokemon.x
    dz = cam_z - pokemon.z
    angle_radians = math.atan2(dx, dz)
    angle_degrees = math.degrees(angle_radians)
    
    glRotatef(angle_degrees, 0, 1, 0)
    
    width = 1.0
    height = 0.15
    hp_ratio = max(0.0, pokemon.hp / pokemon.max_hp)
    
    glDisable(GL_LIGHTING)
    
    # Background (Red)
    glColor3f(1, 0, 0)
    glBegin(GL_QUADS)
    glVertex3f(-width/2, 0, 0); glVertex3f(width/2, 0, 0)
    glVertex3f(width/2, height, 0); glVertex3f(-width/2, height, 0)
    glEnd()
    
    # Foreground (Green)
    z_offset = 0.05
    if hp_ratio > 0:
        glColor3f(0, 1, 0)
        current_width = width * hp_ratio
        glBegin(GL_QUADS)
        glVertex3f(-width/2, 0, z_offset); glVertex3f(-width/2 + current_width, 0, z_offset)
        glVertex3f(-width/2 + current_width, height, z_offset); glVertex3f(-width/2, height, z_offset)
        glEnd()

    glEnable(GL_LIGHTING)
    glPopMatrix()

def build_observation(me, all_pokemons):
    """
    Reconstructs the Observation Vector (Self + 16 Rays).
    Must match _get_obs from custom_env.py exactly.
    """
    # --- Part 1: Self State (2 values) ---
    hp_norm = me.hp / me.max_hp
    # Check cooldown
    cd_norm = me.attack_timer / config.ATTACK_DURATION if me.attack_timer > 0 else 0.0
    self_state = [hp_norm, cd_norm]

    # --- Part 2: Lidar Rays (48 values) ---
    lidar_data = []
    angle_step = 360.0 / config.NUM_RAYS
    max_dist = config.VISION_RANGE
    boundary = config.BOUNDARY

    for i in range(config.NUM_RAYS):
        # Ray Angle relative to agent
        ray_angle = (me.angle + (i * angle_step)) % 360
        rad = math.radians(ray_angle)
        dir_x = math.sin(rad)
        dir_z = math.cos(rad)

        # A. Wall Check
        dist_wall = max_dist
        if abs(dir_x) > 1e-6:
            t1 = (boundary - me.x) / dir_x
            t2 = (-boundary - me.x) / dir_x
            if t1 > 0: dist_wall = min(dist_wall, t1)
            if t2 > 0: dist_wall = min(dist_wall, t2)
        if abs(dir_z) > 1e-6:
            t1 = (boundary - me.z) / dir_z
            t2 = (-boundary - me.z) / dir_z
            if t1 > 0: dist_wall = min(dist_wall, t1)
            if t2 > 0: dist_wall = min(dist_wall, t2)

        # B. Enemy Check
        dist_enemy = max_dist
        hit_enemy = False
        
        for target in all_pokemons:
            if target == me or target.hp <= 0: continue
            
            # Vector to target
            fc_x = target.x - me.x
            fc_z = target.z - me.z
            
            # Project onto ray
            t_proj = fc_x * dir_x + fc_z * dir_z
            
            closest_x = me.x + dir_x * t_proj
            closest_z = me.z + dir_z * t_proj
            
            dist_sq = (closest_x - target.x)**2 + (closest_z - target.z)**2
            radius_sq = config.HITBOX_RADIUS**2
            
            if dist_sq < radius_sq and t_proj > 0:
                offset = math.sqrt(radius_sq - dist_sq)
                t_hit = t_proj - offset
                if t_hit < dist_enemy:
                    dist_enemy = t_hit
                    hit_enemy = True

        # C. Resolve
        final_dist = min(dist_wall, dist_enemy)
        final_dist = min(final_dist, max_dist)
        
        norm_dist = final_dist / max_dist
        is_wall = 1.0 if (final_dist == dist_wall and final_dist < max_dist) else 0.0
        is_enemy = 1.0 if (final_dist == dist_enemy and hit_enemy and final_dist < max_dist) else 0.0
        
        lidar_data.extend([norm_dist, is_wall, is_enemy])

    return np.array(self_state + lidar_data, dtype=np.float32)

def get_action_mask(pokemon, all_pokemons):
    """
    Reconstructs the Action Mask.
    """
    mask = np.zeros(6, dtype=np.int8)
    if pokemon.is_attacking:
        mask[0] = 1 # Only No-Op allowed
    else:
        mask[0] = 1; mask[1] = 1; mask[2] = 1; mask[3] = 1; mask[4] = 1
        if pokemon.check_hit(all_pokemons):
            mask[5] = 1
        else:
            mask[5] = 0
    return mask

def reset_battle():
    # Generate P1 Position
    x1, z1, rot1 = get_random_spawn([])
    
    # Generate P2 Position (pass P1 to ensure distance)
    x2, z2, rot2 = get_random_spawn([(x1, z1)])
    
    # Create Objects
    p1 = Pokemon('pikachu', pokemon_data.POKEMON_DB['pikachu'], (x1, 0, z1))
    p2 = Pokemon('charmander', pokemon_data.POKEMON_DB['charmander'], (x2, 0, z2))
    
    p1.angle = rot1
    p2.angle = rot2
    
    return [p1, p2]

def main():
    args = parse_args()
    
    # --- SETUP PYGAME ---
    pygame.init()
    pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), DOUBLEBUF | OPENGL)
    
    title = f"Pokemon AI Battle | Model: {os.path.basename(args.model_path)}"
    controls = " | Controls: [1-4] Switch View, [V] Next View"
    pygame.display.set_caption(title + controls)
    
    init_gl()
    
    # --- LOAD MODEL ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model_path} on {device}...")
    
    # Calculate dimensions based on Config
    obs_dim = 2 + (config.NUM_RAYS * 3) # Should be ~50
    action_dim = 6
    
    print(f"Expecting Observation Dim: {obs_dim}")
    agent = PokemonAgent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        agent.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully (Global Step: {checkpoint.get('global_step', 'Unknown')})")
    else:
        print(f"Error: Model not found at {args.model_path}")
        print("Please train the model first using train.py")
        return

    agent.eval() 
    
    # --- INITIALIZE BATTLE ---
    pokemons = reset_battle()
    view_idx = 0
    clock = pygame.time.Clock()
    
    running = True
    game_over_timer = 0
    
    while running:
        # 1. EVENT HANDLING
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_v:
                    view_idx = (view_idx + 1) % len(config.VIEWS)
                if event.key == pygame.K_1: view_idx = 0
                if event.key == pygame.K_2: view_idx = 1
                if event.key == pygame.K_3: view_idx = 2
                if event.key == pygame.K_4: view_idx = 3

        # 2. UPDATE TIMERS (Simulate Delta Time)
        # This is critical for the attack animation and cooldowns
        for p in pokemons:
            p.update_timers(config.DT)

        # 3. AI LOGIC (Get Actions)
        p1, p2 = pokemons[0], pokemons[1]
        
        if p1.hp > 0 and p2.hp > 0:
            with torch.no_grad():
                # Build Observations (Lidar)
                obs_p1 = build_observation(p1, pokemons)
                obs_p2 = build_observation(p2, pokemons)
                
                # Build Masks
                mask_p1 = get_action_mask(p1, pokemons)
                mask_p2 = get_action_mask(p2, pokemons)
                
                # Convert to Tensors
                batch_obs = torch.tensor(np.array([obs_p1, obs_p2]), dtype=torch.float32).to(device)
                batch_masks = torch.tensor(np.array([mask_p1, mask_p2]), dtype=torch.int8).to(device)
                
                # Query Agent
                # We pass the mask to ensure the model knows what it can/cannot do
                actions, _, _, _ = agent.get_action_and_value(batch_obs, action_mask=batch_masks)
                actions = actions.cpu().numpy()
                
                act_p1 = actions[0]
                act_p2 = actions[1]
                
            # Apply Actions (With Logic Locking)
            # P1
            if not p1.is_attacking:
                if act_p1 == 1: p1.move_forward()
                elif act_p1 == 2: p1.move_forward(speed=-config.MOVE_SPEED/2)
                elif act_p1 == 3: p1.rotate(-1)
                elif act_p1 == 4: p1.rotate(1)
                elif act_p1 == 5: p1.attack(pokemons)
            
            # P2
            if not p2.is_attacking:
                if act_p2 == 1: p2.move_forward()
                elif act_p2 == 2: p2.move_forward(speed=-config.MOVE_SPEED/2)
                elif act_p2 == 3: p2.rotate(-1)
                elif act_p2 == 4: p2.rotate(1)
                elif act_p2 == 5: p2.attack(pokemons)

        else:
            # Battle Ended
            game_over_timer += 1
            if game_over_timer > 120: 
                pokemons = reset_battle()
                game_over_timer = 0
                print("Battle Reset!")

        # 4. RENDER
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # cam = config.VIEWS[view_idx]
        cam=[0,12,12,0,0,0]
        gluLookAt(cam[0], cam[1], cam[2], 0, 1, 0, 0, 1, 0)
        # gluLookAt(0, 12, 12, 0, 1, 0, 0, 1, 0)
        
        draw_ground()
        draw_walls()
        
        for p in pokemons:
            if p.hp > 0:
                p.draw()
                draw_hp_bar(p, cam)

        pygame.display.flip()
        clock.tick(config.FPS)

    pygame.quit()

if __name__ == "__main__":
    main()