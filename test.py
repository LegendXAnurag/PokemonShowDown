import argparse
import os
import math
import numpy as np
import torch
import time
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

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Trained Pokemon Agents")
    parser.add_argument("--model-path", type=str, default=f"{config.CHECKPOINT_DIR}/no_bs_no_crit_v2_exec_backstab.pt",
        help="path to the trained model checkpoint")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    return parser.parse_args()

def get_n_spawns(n):
    """
    Generates N random spawn positions ensuring MIN_SPAWN_DIST.
    """
    limit = config.BOUNDARY - config.SPAWN_MARGIN
    positions = []
    
    for _ in range(n):
        attempts = 0
        while attempts < 100:
            x = random.uniform(-limit, limit)
            z = random.uniform(-limit, limit)
            valid = True
            for (ex, ez, _) in positions:
                dist = math.sqrt((x - ex)**2 + (z - ez)**2)
                if dist < config.MIN_SPAWN_DIST:
                    valid = False
                    break
            if valid:
                positions.append((x, z, random.uniform(0, 360)))
                break
            attempts += 1
        if len(positions) < (_ + 1):
             positions.append((random.uniform(-limit, limit), random.uniform(-limit, limit), 0))
    return positions

def init_gl():
    glClearColor(0.53, 0.81, 0.92, 1.0) 
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (10, 15, 10, 0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1.0))
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (config.SCREEN_WIDTH / config.SCREEN_HEIGHT), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

def draw_hp_bar(pokemon, camera_pos):
    glPushMatrix()
    glTranslatef(pokemon.x, pokemon.y + 2.0, pokemon.z)
    
    cam_x, cam_y, cam_z = camera_pos
    dx = cam_x - pokemon.x
    dz = cam_z - pokemon.z
    angle_radians = math.atan2(dx, dz)
    angle_degrees = math.degrees(angle_radians)
    
    glRotatef(angle_degrees, 0, 1, 0)
    
    width = 1.0
    height = 0.15
    hp_ratio = max(0.0, pokemon.hp / pokemon.max_hp)
    
    glDisable(GL_LIGHTING)
    glColor3f(1, 0, 0)
    glBegin(GL_QUADS)
    glVertex3f(-width/2, 0, 0); glVertex3f(width/2, 0, 0)
    glVertex3f(width/2, height, 0); glVertex3f(-width/2, height, 0)
    glEnd()
    
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

# --- REPLICATE OBSERVATION LOGIC FROM CUSTOM_ENV ---
def build_observation(me, all_pokemons):
    hp_norm = me.hp / me.max_hp
    cd_norm = me.attack_timer / config.ATTACK_DURATION if me.attack_timer > 0 else 0.0
    self_state = [hp_norm, cd_norm]

    lidar_data = []
    angle_step = 360.0 / config.NUM_RAYS
    max_dist = config.VISION_RANGE
    boundary = config.BOUNDARY

    for i in range(config.NUM_RAYS):
        ray_angle = (me.angle + (i * angle_step)) % 360
        rad = math.radians(ray_angle)
        dir_x = math.sin(rad)
        dir_z = math.cos(rad)

        # 1. Wall Check
        dist_wall = max_dist
        if abs(dir_x) > 1e-6:
            t1 = (boundary - me.x) / dir_x; t2 = (-boundary - me.x) / dir_x
            if t1 > 0: dist_wall = min(dist_wall, t1)
            if t2 > 0: dist_wall = min(dist_wall, t2)
        if abs(dir_z) > 1e-6:
            t1 = (boundary - me.z) / dir_z; t2 = (-boundary - me.z) / dir_z
            if t1 > 0: dist_wall = min(dist_wall, t1)
            if t2 > 0: dist_wall = min(dist_wall, t2)

        # 2. Enemy Check
        dist_enemy = max_dist
        hit_enemy = False
        enemy_obj = None
        
        for target in all_pokemons:
            if target == me or target.hp <= 0: continue
            
            fc_x = target.x - me.x
            fc_z = target.z - me.z
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
                    enemy_obj = target

        final_dist = min(dist_wall, dist_enemy)
        final_dist = min(final_dist, max_dist)
        norm_dist = final_dist / max_dist

        # [CHANGE] Construct 5-Channel Data
        ch_dist = norm_dist
        ch_is_wall = 1.0 if (final_dist == dist_wall and final_dist < max_dist) else 0.0
        ch_is_enemy = 1.0 if (final_dist == dist_enemy and hit_enemy and final_dist < max_dist) else 0.0
        ch_enemy_hp = 0.0
        ch_enemy_face = 0.0

        if hit_enemy and enemy_obj is not None:
            ch_enemy_hp = enemy_obj.hp / enemy_obj.max_hp
            
            # Calculate Facing Angle (Dot Product)
            # Ray Vector (Me -> Enemy)
            ray_dx = dir_x
            ray_dz = dir_z
            
            # Enemy Forward Vector
            # Note: accessing .angle directly or using helper if available
            # We implemented get_forward_vector in pokemon.py
            en_fx, en_fz = enemy_obj.get_forward_vector()
            
            dot = (ray_dx * en_fx) + (ray_dz * en_fz)
            ch_enemy_face = dot

        lidar_data.extend([ch_dist, ch_is_wall, ch_is_enemy, ch_enemy_hp, ch_enemy_face])

    return np.array(self_state + lidar_data, dtype=np.float32)

def get_action_mask(pokemon, all_pokemons):
    mask = np.zeros(6, dtype=np.int8)
    if pokemon.hp <= 0:
        mask[0] = 1 # Dead
    elif pokemon.is_attacking:
        mask[0] = 1 # Locked
    else:
        mask[0] = 1; mask[1] = 1; mask[2] = 1; mask[3] = 1; mask[4] = 1
        if pokemon.check_hit(all_pokemons):
            mask[5] = 1
        else:
            mask[5] = 0
    return mask

def reset_battle():
    spawns = get_n_spawns(config.NUM_AGENTS)
    pokemons = []
    
    species_list = list(pokemon_data.POKEMON_DB.keys())
    
    for i in range(config.NUM_AGENTS):
        species = species_list[i % len(species_list)]
        x, z, rot = spawns[i]
        p = Pokemon(species, pokemon_data.POKEMON_DB[species], (x, 0, z))
        p.angle = rot
        pokemons.append(p)
        
    return pokemons

def main():
    args = parse_args()
    pygame.init()
    pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption(f"N-Agent Battle ({config.NUM_AGENTS} agents) | [SPACE] Reset")
    
    init_gl()
    
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    
    # [CHANGE] Updated Dimension Calculation
    obs_dim = 2 + (config.NUM_RAYS * config.LIDAR_CHANNELS)
    action_dim = 6
    agent = PokemonAgent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        agent.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {args.model_path}")
    else:
        print(f"Model not found at {args.model_path}. Using random weights.")

    agent.eval() 
    
    pokemons = reset_battle()
    clock = pygame.time.Clock()
    running = True
    
    # Camera settings
    cam_angle = 0
    cam_dist = 15
    cam_height = 15

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pokemons = reset_battle()

        # Update Timers
        for p in pokemons:
            p.update_timers(config.DT)

        # AI Inference
        obs_list = []
        mask_list = []
        
        for p in pokemons:
            obs = build_observation(p, pokemons)
            mask = get_action_mask(p, pokemons)
            obs_list.append(obs)
            mask_list.append(mask)

        obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32).to(device)
        mask_tensor = torch.tensor(np.array(mask_list), dtype=torch.int8).to(device)

        with torch.no_grad():
            actions, _, _, _ = agent.get_action_and_value(obs_tensor, action_mask=mask_tensor)
            actions = actions.cpu().numpy()

        # Apply Actions
        alive_count = 0
        for i, p in enumerate(pokemons):
            if p.hp > 0:
                alive_count += 1
                act = actions[i]
                if not p.is_attacking:
                    if act == 1: p.move_forward()
                    elif act == 2: p.move_forward(speed=-config.MOVE_SPEED/2)
                    elif act == 3: p.rotate(-1)
                    elif act == 4: p.rotate(1)
                    elif act == 5: p.attack(pokemons)

        # Auto-Reset if 1 or 0 left
        if alive_count <= 1:
            time.sleep(1)
            pokemons = reset_battle()

        # Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Simple orbiting camera
        # cam_angle += 0.2
        rad = math.radians(cam_angle)
        cx = math.sin(rad) * cam_dist
        cz = math.cos(rad) * cam_dist
        gluLookAt(0, cam_height, cam_dist, 0, 0, 0, 0, 1, 0)
        
        draw_ground()
        draw_walls()
        
        for p in pokemons:
            if p.hp > 0:
                p.draw()
                draw_hp_bar(p, (0, cam_height, cam_dist))

        pygame.display.flip()
        clock.tick(config.FPS)

    pygame.quit()

if __name__ == "__main__":
    main()