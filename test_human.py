# test_human.py
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

# --- ARGS ---
def parse_args():
    parser = argparse.ArgumentParser(description="Human vs AI Pokemon Battle")
    parser.add_argument("--model-path", type=str, default=f"{config.CHECKPOINT_DIR}/{config.MODEL_NAME}",
        help="path to the trained model checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args()

# --- HELPER: Spawns (Fixed Overlap) ---
def get_n_spawns(n):
    limit = config.BOUNDARY - config.SPAWN_MARGIN
    positions = []
    
    # Collision radius for spawn (slightly larger than hitbox to be safe)
    spawn_radius = config.HITBOX_RADIUS * 2.5 
    
    for i in range(n):
        spawned = False
        attempts = 0
        while attempts < 500: # Increased attempts for robustness
            x = random.uniform(-limit, limit)
            z = random.uniform(-limit, limit)
            valid = True
            
            # Check against existing positions
            for (ex, ez, _) in positions:
                dist = math.sqrt((x - ex)**2 + (z - ez)**2)
                # Ensure they are at least spawn_radius apart
                if dist < spawn_radius:
                    valid = False
                    break
            
            if valid:
                positions.append((x, z, random.uniform(0, 360)))
                spawned = True
                break
            attempts += 1
            
        if not spawned:
            print(f"Warning: Forced spawn for agent {i} (possible overlap)")
            positions.append((random.uniform(-limit, limit), random.uniform(-limit, limit), 0))
            
    return positions

# --- HELPER: Init OpenGL ---
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

# --- HELPER: Mouse Raycasting ---
def get_mouse_ground_intersection(mouse_x, mouse_y):
    """
    Unprojects the mouse coordinates to find where the ray intersects the ground plane (y=0).
    """
    # Get current OpenGL matrices
    model_view = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    viewport = glGetIntegerv(GL_VIEWPORT)

    # OpenGL y-coordinate is inverted relative to Pygame
    gl_y = viewport[3] - mouse_y

    try:
        # Unproject two points to get a ray (Near plane z=0, Far plane z=1)
        near_point = gluUnProject(mouse_x, gl_y, 0.0, model_view, projection, viewport)
        far_point = gluUnProject(mouse_x, gl_y, 1.0, model_view, projection, viewport)
    except:
        return None

    ray_origin = np.array(near_point)
    ray_dir = np.array(far_point) - ray_origin
    
    # We want intersection with plane y = 0
    if abs(ray_dir[1]) < 1e-6: # Parallel to ground
        return None
        
    t = -ray_origin[1] / ray_dir[1]
    
    # If intersection is behind camera
    if t < 0:
        return None
        
    intersection = ray_origin + t * ray_dir
    return (intersection[0], intersection[2]) # Return X, Z

# --- HELPER: UI / Indicators ---
def draw_indicators(pokemon, camera_pos, is_player=False):
    glPushMatrix()
    glTranslatef(pokemon.x, pokemon.y + 2.5, pokemon.z)
    
    cam_x, cam_y, cam_z = camera_pos
    dx = cam_x - pokemon.x
    dz = cam_z - pokemon.z
    angle_radians = math.atan2(dx, dz)
    angle_degrees = math.degrees(angle_radians)
    glRotatef(angle_degrees, 0, 1, 0)
    
    # 1. Draw Player Marker
    if is_player:
        glPushMatrix()
        glTranslatef(0, 0.6, 0)
        glScalef(0.4, 0.4, 0.4)
        glDisable(GL_LIGHTING)
        glColor3f(0.2, 1.0, 0.2) 
        glBegin(GL_TRIANGLES)
        glVertex3f(-0.5, 0.5, 0)
        glVertex3f(0.5, 0.5, 0)
        glVertex3f(0, -0.5, 0)
        glEnd()
        glEnable(GL_LIGHTING)
        glPopMatrix()

    # 2. Draw HP Bar
    width = 1.0
    height = 0.15
    hp_ratio = max(0.0, pokemon.hp / pokemon.max_hp)
    
    glDisable(GL_LIGHTING)

    # A. Draw Black Border
    glColor3f(0, 0, 0) # Black
    glLineWidth(2.0)
    glBegin(GL_LINE_LOOP)
    glVertex3f(-width/2, 0, 0)
    glVertex3f(width/2, 0, 0)
    glVertex3f(width/2, height, 0)
    glVertex3f(-width/2, height, 0)
    glEnd()
    glLineWidth(1.0)
    
    # B. Draw Health (Very Dark Green)
    if hp_ratio > 0:
        glColor3f(0.0, 0.35, 0.0) # Very Dark Green
        current_width = width * hp_ratio
        
        glBegin(GL_QUADS)
        # Add slight z-offset? Not needed if we drew outline first, but usually safe
        # However, since outline is lines and this is filled, drawing at same Z is fine 
        # as long as we don't depth fight too much. Let's keep Z=0.
        glVertex3f(-width/2, 0, 0)
        glVertex3f(-width/2 + current_width, 0, 0)
        glVertex3f(-width/2 + current_width, height, 0)
        glVertex3f(-width/2, height, 0)
        glEnd()

    glEnable(GL_LIGHTING)
    glPopMatrix()

# --- AI HELPERS ---
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

        dist_wall = max_dist
        if abs(dir_x) > 1e-6:
            t1 = (boundary - me.x) / dir_x; t2 = (-boundary - me.x) / dir_x
            if t1 > 0: dist_wall = min(dist_wall, t1)
            if t2 > 0: dist_wall = min(dist_wall, t2)
        if abs(dir_z) > 1e-6:
            t1 = (boundary - me.z) / dir_z; t2 = (-boundary - me.z) / dir_z
            if t1 > 0: dist_wall = min(dist_wall, t1)
            if t2 > 0: dist_wall = min(dist_wall, t2)

        dist_entity = max_dist
        hit_entity = False
        entity_obj = None
        
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
                if t_hit < dist_entity:
                    dist_entity = t_hit
                    hit_entity = True
                    entity_obj = target

        final_dist = min(dist_wall, dist_entity)
        final_dist = min(final_dist, max_dist)
        norm_dist = final_dist / max_dist

        ch_dist = norm_dist
        ch_is_wall = 1.0 if (final_dist == dist_wall and final_dist < max_dist) else 0.0
        ch_is_enemy = 0.0
        ch_is_teammate = 0.0
        ch_unit_hp = 0.0
        ch_unit_face = 0.0

        if hit_entity and entity_obj is not None and final_dist < dist_wall:
            if entity_obj.team_id == me.team_id:
                ch_is_teammate = 1.0
            else:
                ch_is_enemy = 1.0
            ch_unit_hp = entity_obj.hp / entity_obj.max_hp
            ray_dx = dir_x; ray_dz = dir_z
            en_fx, en_fz = entity_obj.get_forward_vector()
            dot = (ray_dx * en_fx) + (ray_dz * en_fz)
            ch_unit_face = dot

        lidar_data.extend([ch_dist, ch_is_wall, ch_is_enemy, ch_is_teammate, ch_unit_hp, ch_unit_face])

    return np.array(self_state + lidar_data, dtype=np.float32)

def check_collision(me, x, z, all_mons):
    # Collision threshold (radius + radius)
    min_dist_sq = 1.0**2
    for target in all_mons:
        if target == me or target.hp <= 0: continue
        dist_sq = (x - target.x)**2 + (z - target.z)**2
        if dist_sq < min_dist_sq:
            return True
    return False

def get_action_mask(pokemon, all_pokemons):
    mask = np.zeros(6, dtype=np.int8)
    if pokemon.hp <= 0:
        mask[0] = 1 
    elif pokemon.is_attacking:
        mask[0] = 1 
    else:
        mask[0] = 0; mask[1] = 1; mask[2] = 1 if config.ALLOW_BACKWARD else 0;
        mask[3] = 1; mask[4] = 1
        mask[5] = 1 if pokemon.check_hit(all_pokemons) else 0
        
        px, pz = pokemon.predict_position(1)
        if check_collision(pokemon, px, pz, all_pokemons):
            mask[1] = 0
            
        px, pz = pokemon.predict_position(-1)
        if check_collision(pokemon, px, pz, all_pokemons):
            mask[2] = 0
            
    return mask

def reset_battle():
    total_agents = config.NUM_AGENTS
    spawns = get_n_spawns(total_agents)
    pokemons = []
    
    species_list = list(pokemon_data.POKEMON_DB.keys())
    
    global_idx = 0
    player_assigned = False

    for team_idx, team_setup in enumerate(config.TEAMS_SETUP):
        count = team_setup["count"]
        color = team_setup["color"]
        
        for _ in range(count):
            if global_idx >= len(spawns): break
            species = random.choice(species_list)
            x, z, rot = spawns[global_idx]
            
            p = Pokemon(species, pokemon_data.POKEMON_DB[species], (x, 0, z), team_idx, color)
            p.angle = rot
            
            if team_idx == 0 and not player_assigned:
                player_assigned = True
                p.is_human = True
            else:
                p.is_human = False
                
            pokemons.append(p)
            global_idx += 1
        
    return pokemons

def main():
    args = parse_args()
    
    pygame.init()
    pygame.font.init()
    pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption(f"YOU (Team 0) vs AI | WASD + Mouse + Space/Click")
    
    init_gl()
    
    device = torch.device("cpu")
    print(f"Loading model on {device}...")
    
    obs_dim = 2 + (config.NUM_RAYS * config.LIDAR_CHANNELS)
    action_dim = 6
    agent = PokemonAgent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        agent.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {args.model_path}")
    else:
        print(f"Model not found at {args.model_path}. AI will act randomly.")

    agent.eval() 
    
    random.seed(args.seed)
    pokemons = reset_battle()
    clock = pygame.time.Clock()
    running = True
    game_over = False
    
    while running:
        # A. Event Handling
        attack_command = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and game_over:
                    pokemons = reset_battle()
                    game_over = False
                if event.key == pygame.K_SPACE:
                    attack_command = True
            
            # Attack on Left Click
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left Click
                    attack_command = True

        player_p = None
        for p in pokemons:
            if getattr(p, 'is_human', False):
                player_p = p
                break
        
        for p in pokemons:
            p.update_timers(config.DT)

        # B. RENDER & CAMERA SETUP (First Pass for Camera)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        cam_x, cam_y, cam_z = 0, 15, 15
        if player_p:
            target_x = player_p.x
            target_z = player_p.z + 10
            cam_x, cam_y, cam_z = target_x, 10, target_z + 8
            gluLookAt(cam_x, cam_y, cam_z, player_p.x, 0, player_p.z, 0, 1, 0)
        else:
            gluLookAt(0, 25, 20, 0, 0, 0, 0, 1, 0)

        # C. GAME LOGIC
        if not game_over:
            # 1. Human Logic
            if player_p and player_p.hp > 0:
                if not player_p.is_attacking:
                    if attack_command:
                        player_p.attack(pokemons)
                    else:
                        # Mouse Rotation
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        ground_pos = get_mouse_ground_intersection(mouse_x, mouse_y)
                        
                        if ground_pos:
                            mx, mz = ground_pos
                            dx = mx - player_p.x
                            dz = mz - player_p.z
                            
                            # Update angle if mouse is not exactly on top of player
                            if (dx*dx + dz*dz) > 0.01:
                                target_angle = math.degrees(math.atan2(dx, dz))
                                player_p.angle = target_angle

                        # Keyboard Movement
                        keys = pygame.key.get_pressed()
                        
                        # Rotation overrides (Optional mixing)
                        if keys[pygame.K_a]: player_p.rotate(-1)
                        if keys[pygame.K_d]: player_p.rotate(1)
                        
                        # Forward Movement with Collision Check
                        if keys[pygame.K_w]:
                            # Predict position
                            pred_x, pred_z = player_p.predict_position(1)
                            # Only move if no collision
                            if not check_collision(player_p, pred_x, pred_z, pokemons):
                                player_p.move_forward()

            # 2. AI Logic
            ai_indices = []
            obs_list = []
            mask_list = []
            
            for i, p in enumerate(pokemons):
                if getattr(p, 'is_human', False): continue
                if p.hp <= 0: continue
                
                obs = build_observation(p, pokemons)
                mask = get_action_mask(p, pokemons)
                
                obs_list.append(obs)
                mask_list.append(mask)
                ai_indices.append(i)

            if obs_list:
                obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32).to(device)
                mask_tensor = torch.tensor(np.array(mask_list), dtype=torch.int8).to(device)

                with torch.no_grad():
                    actions, _, _, _ = agent.get_action_and_value(obs_tensor, action_mask=mask_tensor)
                    actions = actions.cpu().numpy()

                for idx, act in zip(ai_indices, actions):
                    p = pokemons[idx]
                    if not p.is_attacking:
                        if act == 1: p.move_forward()
                        elif act == 2: p.move_forward(speed=-config.MOVE_SPEED/2)
                        elif act == 3: p.rotate(-1)
                        elif act == 4: p.rotate(1)
                        elif act == 5: p.attack(pokemons)

        # D. Win Condition
        alive_teams = set()
        for p in pokemons:
            if p.hp > 0:
                alive_teams.add(p.team_id)

        if not game_over:
            if len(config.TEAMS_SETUP) > 1:
                if len(alive_teams) <= 1: game_over = True
            else:
                if len(alive_teams) == 0: game_over = True
            
            if game_over:
                if len(alive_teams) == 1:
                    w_id = list(alive_teams)[0]
                    if w_id == 0: caption = "YOU WON! (Press R)"
                    else: caption = f"TEAM {w_id} WON! (Press R)"
                else: caption = "DRAW! (Press R)"
                pygame.display.set_caption(caption)
                print(f"Game Over. {caption}")

        # E. Final Draw
        draw_ground()
        draw_walls()
        
        for p in pokemons:
            if p.hp > 0:
                p.draw()
                is_user = getattr(p, 'is_human', False)
                draw_indicators(p, (cam_x, cam_y, cam_z), is_player=is_user)

        pygame.display.flip()
        clock.tick(config.FPS)

    pygame.quit()

if __name__ == "__main__":
    main()