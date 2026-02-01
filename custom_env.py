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
        "name": "pokemon_battle_lidar_n_v0",
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        # [CHANGE] Dynamic Agent Names
        self.possible_agents = [f"agent_{i}" for i in range(config.NUM_AGENTS)]
        self.agents = self.possible_agents[:]
        
        self.action_spaces = {agent: spaces.Discrete(6) for agent in self.possible_agents}

        # Obs: Self(2) + Lidar(Num_Rays * 3)
        obs_dim = 2 + (config.NUM_RAYS * 3)
        
        self.observation_spaces = {
            agent: spaces.Dict({
                "observation": spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32),
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

        # [CHANGE] N-Agent Spawn Logic
        limit = config.BOUNDARY - config.SPAWN_MARGIN
        positions = []
        
        for _ in range(config.NUM_AGENTS):
            attempts = 0
            while attempts < 100:
                x = random.uniform(-limit, limit)
                z = random.uniform(-limit, limit)
                valid = True
                for (ex, ez) in positions:
                    dist = math.sqrt((x - ex)**2 + (z - ez)**2)
                    if dist < config.MIN_SPAWN_DIST:
                        valid = False
                        break
                if valid:
                    positions.append((x, z))
                    break
                attempts += 1
            if len(positions) < (_ + 1):
                # Fallback if too crowded: just spawn somewhere random
                positions.append((random.uniform(-limit, limit), random.uniform(-limit, limit)))

        # Create Instances (Randomly assign species)
        species_list = list(pokemon_data.POKEMON_DB.keys())
        self.pokemon_instances = {}
        
        for i, agent_id in enumerate(self.agents):
            species = random.choice(species_list)
            pos = (positions[i][0], 0, positions[i][1])
            p = Pokemon(species, pokemon_data.POKEMON_DB[species], pos)
            p.angle = random.uniform(0, 360)
            self.pokemon_instances[agent_id] = p

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        self.steps_count += 1
        
        # 1. Update Timers (Resets damage dealt counters)
        for p in self.pokemon_instances.values():
            p.update_timers(config.DT)

        prev_hp = {name: p.hp for name, p in self.pokemon_instances.items()}
        all_mons = list(self.pokemon_instances.values())

        # 2. Execute Actions
        for agent in self.agents:
            if agent not in actions: continue
            
            pokemon = self.pokemon_instances[agent]
            action = actions[agent]
            
            # Dead agents do nothing
            if pokemon.hp <= 0: continue

            if pokemon.is_attacking:
                pass 
            else:
                if action == 1: pokemon.move_forward()
                elif action == 2: pokemon.move_forward(speed=-config.MOVE_SPEED/2)
                elif action == 3: pokemon.rotate(-1)
                elif action == 4: pokemon.rotate(1)
                elif action == 5: pokemon.attack(all_mons) # Uses new simplified attack logic

        # 3. Rewards & Termination
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        observations = {}
        
        alive_count = sum(1 for p in self.pokemon_instances.values() if p.hp > 0)
        
        # [CHANGE] Termination: End if 0 or 1 survivor remains
        game_over = (alive_count <= 1)
        is_timeout = self.steps_count >= config.MAX_STEPS_PER_EPISODE

        for agent in self.agents:
            pokemon = self.pokemon_instances[agent]
            
            damage_taken = prev_hp[agent] - pokemon.hp
            
            # [CHANGE] Reward calculation based on self.damage_dealt_this_step
            reward = config.REWARD_STEP_PENALTY
            reward += (pokemon.damage_dealt_this_step * config.REWARD_DMG_DEALT_SCALE)
            reward -= (damage_taken * config.REWARD_DMG_TAKEN_SCALE)
            
            # Win/Loss
            if game_over:
                if pokemon.hp > 0:
                    reward += config.REWARD_WIN
                else:
                    # Only apply loss penalty if they died THIS step or were already dead
                    # (Usually better to apply large penalty at moment of death)
                    if prev_hp[agent] > 0 and pokemon.hp <= 0:
                         reward += config.REWARD_LOSS
            
            # Death penalty moment
            if prev_hp[agent] > 0 and pokemon.hp <= 0 and not game_over:
                 reward += config.REWARD_LOSS

            rewards[agent] = reward
            terminations[agent] = game_over
            truncations[agent] = is_timeout
            infos[agent] = {"hp": pokemon.hp}
            observations[agent] = self._get_obs(agent)
            
        if game_over or is_timeout:
            self.agents = [] 

        return observations, rewards, terminations, truncations, infos

    def _get_obs(self, agent):
        me = self.pokemon_instances[agent]
        all_mons = list(self.pokemon_instances.values())
        
        # Self State
        hp_norm = me.hp / me.max_hp
        cd_norm = me.attack_timer / config.ATTACK_DURATION if me.attack_timer > 0 else 0.0
        self_state = [hp_norm, cd_norm]

        # Lidar
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

            # B. Enemies
            dist_enemy = max_dist
            hit_enemy = False
            
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
                    if t_hit < dist_enemy:
                        dist_enemy = t_hit
                        hit_enemy = True

            final_dist = min(dist_wall, dist_enemy)
            final_dist = min(final_dist, max_dist)
            
            norm_dist = final_dist / max_dist
            is_wall = 1.0 if (final_dist == dist_wall and final_dist < max_dist) else 0.0
            is_enemy = 1.0 if (final_dist == dist_enemy and hit_enemy and final_dist < max_dist) else 0.0
            
            lidar_data.extend([norm_dist, is_wall, is_enemy])

        full_obs = np.array(self_state + lidar_data, dtype=np.float32)

        # Masking
        mask = np.zeros(6, dtype=np.int8)
        
        # Dead agents can only No-Op
        if me.hp <= 0:
            mask[0] = 1
        elif me.is_attacking:
            mask[0] = 1
        else:
            mask[0] = 1; mask[1] = 1; mask[2] = 1; mask[3] = 1; mask[4] = 1
            if me.check_hit(all_mons):
                mask[5] = 1
            else:
                mask[5] = 0

        return {
            "observation": full_obs,
            "action_mask": mask
        }

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
        
        draw_ground()
        draw_walls()
        
        for p in self.pokemon_instances.values():
            if p.hp > 0: p.draw()

        pygame.display.flip()
        if self.clock: self.clock.tick(config.FPS)

    def _init_render(self):
        if self.window is not None: return
        pygame.init()
        self.window = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("Pokemon N-Agent Battle")
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