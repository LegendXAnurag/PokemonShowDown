import functools
import math
import numpy as np
import random  # <--- Added import
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
        "name": "pokemon_battle_lidar_v0",
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.possible_agents = ["pikachu", "charmander"]
        self.agents = self.possible_agents[:]
        
        self.action_spaces = {agent: spaces.Discrete(6) for agent in self.possible_agents}

        # [CHANGE] Calculate Observation Size
        # Self Features: HP (1) + AttackTimer (1) = 2
        # Lidar Features: Rays (16) * Features (3: Dist, IsWall, IsEnemy) = 48
        # Total = 50
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
        self.steps_count = 0 # [CHANGE] Track episode steps

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.steps_count = 0 # [CHANGE] Reset timeout counter
        
        # Set seed for reproducibility if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # --- RANDOMIZED SPAWN LOGIC ---
        limit = config.BOUNDARY - config.SPAWN_MARGIN
        
        # 1. Generate Position for Agent 1 (Pikachu)
        x1 = random.uniform(-limit, limit)
        z1 = random.uniform(-limit, limit)
        rot1 = random.uniform(0, 360)
        
        # 2. Generate Position for Agent 2 (Charmander) with Minimum Distance Check
        while True:
            x2 = random.uniform(-limit, limit)
            z2 = random.uniform(-limit, limit)
            dist = math.sqrt((x1 - x2)**2 + (z1 - z2)**2)
            if dist >= config.MIN_SPAWN_DIST:
                break
        rot2 = random.uniform(0, 360)
        
        # 3. Create Instances
        self.pokemon_instances = {
            "pikachu": Pokemon('pikachu', pokemon_data.POKEMON_DB['pikachu'], (x1, 0, z1)),
            "charmander": Pokemon('charmander', pokemon_data.POKEMON_DB['charmander'], (x2, 0, z2))
        }
        self.pokemon_instances["pikachu"].angle = rot1
        self.pokemon_instances["charmander"].angle = rot2

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        # [CHANGE] 1. Update Global Timers and Step Count
        self.steps_count += 1
        for p in self.pokemon_instances.values():
            p.update_timers(config.DT)

        all_mons = list(self.pokemon_instances.values())
        prev_hp = {name: p.hp for name, p in self.pokemon_instances.items()}

        for agent in self.agents:
            if agent not in actions: continue
            
            action = actions[agent]
            pokemon = self.pokemon_instances[agent]
            
            if pokemon.hp <= 0: continue

            # [CHANGE] Action Locking Logic
            # If attacking, force No-Op (Action 0).
            # The Action Mask should have prevented this, but we enforce it here too.
            if pokemon.is_attacking:
                pass # Locked
            else:
                if action == 1: pokemon.move_forward()
                elif action == 2: pokemon.move_forward(speed=-config.MOVE_SPEED/2)
                elif action == 3: pokemon.rotate(-1)
                elif action == 4: pokemon.rotate(1)
                elif action == 5: pokemon.attack(all_mons)
                # Action 0 is No-Op

        # 2. Rewards & Termination
        rewards = {}
        terminations = {}
        truncations = {} # [CHANGE] For Timeouts
        infos = {}
        observations = {}
        
        game_over = False
        
        # [CHANGE] Check Timeout
        is_timeout = self.steps_count >= config.MAX_STEPS_PER_EPISODE

        for agent in self.agents:
            pokemon = self.pokemon_instances[agent]
            opponent_name = [a for a in self.agents if a != agent][0]
            opponent = self.pokemon_instances[opponent_name]

            damage_dealt = prev_hp[opponent_name] - opponent.hp
            damage_taken = prev_hp[agent] - pokemon.hp
            
            reward = config.REWARD_STEP_PENALTY
            reward += (damage_dealt * config.REWARD_DMG_DEALT_SCALE)
            reward -= (damage_taken * config.REWARD_DMG_TAKEN_SCALE)
            
            # Win/Loss Condition
            if opponent.hp <= 0 and pokemon.hp > 0:
                reward += config.REWARD_WIN
                game_over = True
            elif pokemon.hp <= 0:
                reward += config.REWARD_LOSS
                game_over = True
            
            rewards[agent] = reward
            terminations[agent] = game_over
            truncations[agent] = is_timeout
            infos[agent] = {"hp": pokemon.hp}
            observations[agent] = self._get_obs(agent)
            
        if game_over or is_timeout:
            self.agents = [] 

        return observations, rewards, terminations, truncations, infos

    def _get_obs(self, agent):
        """
        [CHANGE] Generates Self State + 16-Ray Lidar Observation.
        """
        me = self.pokemon_instances[agent]
        all_mons = list(self.pokemon_instances.values())
        
        # --- Part 1: Self State (2 values) ---
        # Normalize HP and Cooldown
        hp_norm = me.hp / me.max_hp
        cd_norm = me.attack_timer / config.ATTACK_DURATION if me.attack_timer > 0 else 0.0
        self_state = [hp_norm, cd_norm]

        # --- Part 2: Lidar Rays (48 values) ---
        lidar_data = []
        angle_step = 360.0 / config.NUM_RAYS
        max_dist = config.VISION_RANGE
        boundary = config.BOUNDARY

        for i in range(config.NUM_RAYS):
            # Calculate ray angle (Egocentric: relative to agent's facing)
            # 0 deg = Front, 90 deg = Right
            ray_angle = (me.angle + (i * angle_step)) % 360
            rad = math.radians(ray_angle)
            dir_x = math.sin(rad)
            dir_z = math.cos(rad)

            # A. Wall Check
            dist_wall = max_dist
            
            # Intersection with X boundaries
            if abs(dir_x) > 1e-6:
                t1 = (boundary - me.x) / dir_x
                t2 = (-boundary - me.x) / dir_x
                # We only care about forward direction (t > 0)
                if t1 > 0: dist_wall = min(dist_wall, t1)
                if t2 > 0: dist_wall = min(dist_wall, t2)
            
            # Intersection with Z boundaries
            if abs(dir_z) > 1e-6:
                t1 = (boundary - me.z) / dir_z
                t2 = (-boundary - me.z) / dir_z
                if t1 > 0: dist_wall = min(dist_wall, t1)
                if t2 > 0: dist_wall = min(dist_wall, t2)

            # B. Enemy/Obstacle Check
            dist_enemy = max_dist
            hit_enemy = False
            
            for target in all_mons:
                if target == me or target.hp <= 0: continue
                
                # Simple Ray-Circle intersection
                # Vector to circle center
                fc_x = target.x - me.x
                fc_z = target.z - me.z
                
                # Project vector onto ray direction
                t_proj = fc_x * dir_x + fc_z * dir_z
                
                # Closest point on ray to center
                closest_x = me.x + dir_x * t_proj
                closest_z = me.z + dir_z * t_proj
                
                # Distance from closest point to center
                dist_sq = (closest_x - target.x)**2 + (closest_z - target.z)**2
                radius_sq = config.HITBOX_RADIUS**2
                
                if dist_sq < radius_sq and t_proj > 0:
                    # Intersection exists. Calculate precise distance.
                    # Offset from closest point back to rim of circle
                    offset = math.sqrt(radius_sq - dist_sq)
                    t_hit = t_proj - offset
                    if t_hit < dist_enemy:
                        dist_enemy = t_hit
                        hit_enemy = True

            # C. Resolve Ray
            final_dist = min(dist_wall, dist_enemy)
            final_dist = min(final_dist, max_dist) # Clamp
            
            # Features: [Normalized Dist, IsWall, IsEnemy]
            norm_dist = final_dist / max_dist
            is_wall = 1.0 if (final_dist == dist_wall and final_dist < max_dist) else 0.0
            is_enemy = 1.0 if (final_dist == dist_enemy and hit_enemy and final_dist < max_dist) else 0.0
            
            lidar_data.extend([norm_dist, is_wall, is_enemy])

        # Combine
        full_obs = np.array(self_state + lidar_data, dtype=np.float32)

        # --- Part 3: Action Masking ---
        mask = np.zeros(6, dtype=np.int8)
        
        if me.is_attacking:
            # LOCKED: Only No-Op allowed
            mask[0] = 1
        else:
            # UNLOCKED
            mask[0] = 1 # NoOp
            mask[1] = 1 # Fwd
            mask[2] = 1 # Bwd
            mask[3] = 1 # Left
            mask[4] = 1 # Right
            
            # Attack logic:
            # Must not be on cooldown (covered by is_attacking check usually, but double check)
            # Must have valid target in sights
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
        gluLookAt(0, 20, 15, 0, 0, 0, 0, 1, 0)
        
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
        pygame.display.set_caption("Pokemon RL Environment")
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