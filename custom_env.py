# custom_env.py
import functools
import math
import numpy as np
import random
from gymnasium import spaces
from pettingzoo import ParallelEnv
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *

import config
import pokemon_data
from pokemon import Pokemon
from ground import draw_ground, draw_walls

class PokemonBattleEnv(ParallelEnv):
    metadata = {
        "name": "pokemon_battle_team_v0",
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        
        self.possible_agents = []
        self.agent_team_map = {}
        
        count = 0
        for team_idx, team_setup in enumerate(config.TEAMS_SETUP):
            num_members = team_setup["count"]
            for _ in range(num_members):
                agent_name = f"agent_{count}"
                self.possible_agents.append(agent_name)
                self.agent_team_map[agent_name] = team_idx
                count += 1
                
        self.agents = self.possible_agents[:]
        self.action_spaces = {agent: spaces.Discrete(6) for agent in self.possible_agents}

        obs_dim = 2 + (config.NUM_RAYS * config.LIDAR_CHANNELS)
        
        self.observation_spaces = {
            agent: spaces.Dict({
                "observation": spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(6,), dtype=np.int8)
            })
            for agent in self.possible_agents
        }

        self.pokemon_instances = {}
        self.window = None
        self.clock = None
        self.steps_count = 0 

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.steps_count = 0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        limit = config.BOUNDARY - config.SPAWN_MARGIN
        positions = []
        # agent_idx is synonymous with index in self.agents
        
        num_teams = len(config.TEAMS_SETUP)
        
        for i, agent_id in enumerate(self.agents):
            team_idx = self.agent_team_map[agent_id]
            
            # Calculate Team Center
            # Place teams in a circle
            radius = (config.BOUNDARY - config.SPAWN_MARGIN) * 0.75 # Use 75% of available space
            angle = (2 * math.pi / num_teams) * team_idx
            cx = radius * math.cos(angle)
            cz = radius * math.sin(angle)
            
            attempts = 0
            placed = False
            while attempts < 100:
                # Spawn around team center
                r = random.uniform(0, config.TEAM_MEMBER_DIST)
                theta = random.uniform(0, 2*math.pi)
                x = cx + r * math.cos(theta)
                z = cz + r * math.sin(theta)
                
                # Check bounds
                if not (-limit <= x <= limit and -limit <= z <= limit):
                    attempts += 1
                    continue
                    
                valid = True
                for j, (ex, ez) in enumerate(positions):
                    existing_agent = self.agents[j]
                    existing_team = self.agent_team_map[existing_agent]
                    dist = math.sqrt((x - ex)**2 + (z - ez)**2)
                    
                    if existing_team == team_idx:
                        # Ensure no overlap with teammate
                        if dist < config.MIN_TEAMMATE_DIST: 
                            valid = False
                            break
                    else:
                        # Ensure distance from enemy
                        if dist < config.MIN_SPAWN_DIST:
                            valid = False
                            break
                
                if valid:
                    positions.append((x, z))
                    placed = True
                    break
                attempts += 1
                
            if not placed:
                # Fallback: Random position
                positions.append((random.uniform(-limit, limit), random.uniform(-limit, limit)))

        species_list = list(pokemon_data.POKEMON_DB.keys())
        self.pokemon_instances = {}
        
        for i, agent_id in enumerate(self.agents):
            species = random.choice(species_list)
            pos = (positions[i][0], 0, positions[i][1])
            
            team_id = self.agent_team_map[agent_id]
            team_color = config.TEAMS_SETUP[team_id]["color"]
            
            p = Pokemon(species, pokemon_data.POKEMON_DB[species], pos, team_id, team_color)
            p.angle = random.uniform(0, 360)
            self.pokemon_instances[agent_id] = p

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        self.steps_count += 1
        
        for p in self.pokemon_instances.values():
            p.update_timers(config.DT)

        prev_hp = {name: p.hp for name, p in self.pokemon_instances.items()}
        all_mons = list(self.pokemon_instances.values())

        for agent in self.agents:
            if agent not in actions: continue
            
            pokemon = self.pokemon_instances[agent]
            action = actions[agent]
            
            if pokemon.hp <= 0: continue

            if pokemon.is_attacking:
                pass 
            else:
                if action == 1: pokemon.move_forward()
                elif action == 2: pokemon.move_forward(speed=-config.MOVE_SPEED/2)
                elif action == 3: pokemon.rotate(-1)
                elif action == 4: pokemon.rotate(1)
                elif action == 5: pokemon.attack(all_mons) 

        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        observations = {}
        
        alive_teams = set()
        for agent_id, p in self.pokemon_instances.items():
            if p.hp > 0:
                alive_teams.add(self.agent_team_map[agent_id])
        
        start_teams = len(config.TEAMS_SETUP)
        if start_teams > 1:
            game_over = (len(alive_teams) <= 1)
        else:
            game_over = (len(alive_teams) == 0)

        is_timeout = self.steps_count >= config.MAX_STEPS_PER_EPISODE

        for agent in self.agents:
            pokemon = self.pokemon_instances[agent]
            team_id = self.agent_team_map[agent]
            
            damage_taken = prev_hp[agent] - pokemon.hp
            
            reward = config.REWARD_STEP_PENALTY
            
            dmg_reward = pokemon.effective_damage_reward * config.REWARD_DMG_DEALT_SCALE
            if pokemon.was_backstab_this_step:
                dmg_reward *= config.REWARD_BACKSTAB_BONUS
            reward += dmg_reward
            
            if pokemon.friendly_fire_damage > 0:
                reward -= (pokemon.friendly_fire_damage * config.REWARD_FRIENDLY_FIRE_SCALE)

            if (pokemon.hp <= 0 and damage_taken>0):
                reward -= config.DEATH_PENALTY

            if damage_taken > 0:
                hp_ratio = pokemon.hp / pokemon.max_hp
                fear_multiplier = 1.0 + (1.0 - hp_ratio) * (config.REWARD_CRITICAL_SCALE - 1.0)
                reward -= (damage_taken * config.REWARD_DMG_TAKEN_SCALE * fear_multiplier)
            
            if game_over:
                if team_id in alive_teams:
                    reward += config.REWARD_WIN
                elif len(alive_teams) > 0:
                    reward += config.REWARD_LOSS
                else:
                    reward += config.REWARD_LOSS

            rewards[agent] = reward
            terminations[agent] = game_over
            truncations[agent] = is_timeout
            infos[agent] = {"hp": pokemon.hp, "team": team_id}
            observations[agent] = self._get_obs(agent)
            
        if is_timeout:
            for agent in self.agents:
                rewards[agent]+= config.TIMEOUT_LOSS

            self.agents = [] 
        if game_over or is_timeout:
            self.agents=[]

        return observations, rewards, terminations, truncations, infos

    def _get_obs(self, agent):
        me = self.pokemon_instances[agent]
        all_mons = list(self.pokemon_instances.values())
        
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

            # A. Wall
            dist_wall = max_dist
            if abs(dir_x) > 1e-6:
                t1 = (boundary - me.x) / dir_x; t2 = (-boundary - me.x) / dir_x
                if t1 > 0: dist_wall = min(dist_wall, t1)
                if t2 > 0: dist_wall = min(dist_wall, t2)
            if abs(dir_z) > 1e-6:
                t1 = (boundary - me.z) / dir_z; t2 = (-boundary - me.z) / dir_z
                if t1 > 0: dist_wall = min(dist_wall, t1)
                if t2 > 0: dist_wall = min(dist_wall, t2)
            if abs(dir_z) > 1e-6:
                t1 = (boundary - me.z) / dir_z; t2 = (-boundary - me.z) / dir_z
                if t1 > 0: dist_wall = min(dist_wall, t1)
                if t2 > 0: dist_wall = min(dist_wall, t2)

            # B. Entities
            dist_entity = max_dist
            hit_entity = False
            entity_obj = None
            
            for target in all_mons:
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

        full_obs = np.array(self_state + lidar_data, dtype=np.float32)

        # [FIX] Action Masking with Collision Detection
        mask = np.zeros(6, dtype=np.int8)
        if me.hp <= 0:
            mask[0] = 1
        elif me.is_attacking:
            mask[0] = 1
        else:
            mask[0] = 0 # No Op
            mask[1] = 1 # Forward (Tentative)
            mask[2] = 1 if config.ALLOW_BACKWARD else 0 # Backward (Tentative)
            mask[3] = 1 # Rotate L
            mask[4] = 1 # Rotate R
            mask[5] = 1 if me.check_hit(all_mons) else 0 # Attack

            # Collision Check for Forward (1)
            pred_x, pred_z = me.predict_position(1)
            if self._check_collision(me, pred_x, pred_z, all_mons):
                mask[1] = 0
            
            # Collision Check for Backward (2)
            if config.ALLOW_BACKWARD:
                pred_x, pred_z = me.predict_position(-1)
                if self._check_collision(me, pred_x, pred_z, all_mons):
                    mask[2] = 0
            else:
                mask[2] = 0

        return {
            "observation": full_obs,
            "action_mask": mask
        }

    def _check_collision(self, me, x, z, all_mons):
        """Returns True if (x,z) collides with any other living pokemon."""
        # Simple Circle Collision
        # Collision dist = radius + radius = 0.5 + 0.5 = 1.0
        min_dist_sq = 1.0**2 
        
        for target in all_mons:
            if target == me or target.hp <= 0: continue
            dist_sq = (x - target.x)**2 + (z - target.z)**2
            if dist_sq < min_dist_sq:
                return True
        return False
    # In custom_env.py

    def render(self):
        if self.render_mode != "human": return
        if self.window is None: self._init_render()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                exit()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        gluLookAt(0, 25, 20, 0, 0, 0, 0, 1, 0)
        
        # 1. Draw Static Environment (Opaque)
        draw_ground()
        draw_walls()
        
        # 2. Draw Pokemon Bodies (Opaque)
        # We draw these first so the Z-buffer is filled with their depth
        for p in self.pokemon_instances.values():
            if p.hp > 0: 
                p.draw_model() # Changed from p.draw()

        # 3. Draw Attack Beams (Transparent)
        # We draw these last. Because we disabled DepthMask in draw_beam,
        # they will visually blend on top of pokemon, but won't delete them.
        for p in self.pokemon_instances.values():
            if p.hp > 0:
                p.draw_beam()

        pygame.display.flip()
        if self.clock: self.clock.tick(config.FPS)
    # def render(self):
    #     if self.render_mode != "human": return
    #     if self.window is None: self._init_render()

    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             self.close()
    #             exit()

    #     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    #     glLoadIdentity()
        
    #     # [FIX] Original Camera Position
    #     gluLookAt(0, 25, 20, 0, 0, 0, 0, 1, 0)
        
    #     draw_ground()
    #     draw_walls()
        
    #     for p in self.pokemon_instances.values():
    #         if p.hp > 0: p.draw()

    #     pygame.display.flip()
    #     if self.clock: self.clock.tick(config.FPS)

    def _init_render(self):
        if self.window is not None: return
        pygame.init()
        self.window = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("Pokemon Team Battle")
        self.clock = pygame.time.Clock()
        glClearColor(0.53, 0.81, 0.92, 1.0) 
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT0, GL_POSITION, (10, 15, 10, 0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.3, 0.3, 0.3, 1.0))
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, (config.SCREEN_WIDTH / config.SCREEN_HEIGHT), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]