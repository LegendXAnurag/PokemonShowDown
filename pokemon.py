# pokemon.py
import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import config
from utils import ModelLoader

class Pokemon:
    def __init__(self, name, data, start_pos):
        self.name = name
        self.hp = data['hp']
        self.max_hp = data['hp']
        self.attack_power = data['attack_power']
        self.color = data['color_fallback']
        rot_offset = data.get('rotation_correction', 0)
        self.model = ModelLoader(data['model_path'], data['color_fallback'], rotation_offset=rot_offset)
        
        self.x = start_pos[0]
        self.y = (config.POKEMON_SCALE_SIZE / 2.0)
        self.z = start_pos[2]
        self.angle = 0.0 
        
        self.attack_timer = 0.0 
        self.is_attacking = False
        self.actual_beam_length = 0.0
        
        # Reward tracking
        self.damage_dealt_this_step = 0.0
        self.effective_damage_reward = 0.0 # [CHANGE] Tracks damage weighted by Execute Bonus
        self.was_backstab_this_step = False 

    def update_timers(self, dt):
        self.damage_dealt_this_step = 0.0 
        self.effective_damage_reward = 0.0 # [CHANGE] Reset
        self.was_backstab_this_step = False
        if self.attack_timer > 0:
            self.attack_timer -= dt
            self.is_attacking = True
        else:
            self.attack_timer = 0
            self.is_attacking = False

    def get_forward_vector(self):
        rad = math.radians(self.angle)
        return (math.sin(rad), math.cos(rad))

    def move_forward(self, speed=None):
        if self.is_attacking: return

        if speed is None: speed = config.MOVE_SPEED
        rad = math.radians(self.angle)
        dx = math.sin(rad) * speed
        dz = math.cos(rad) * speed
        new_x = self.x + dx
        new_z = self.z + dz
        limit = config.BOUNDARY - (config.POKEMON_SCALE_SIZE / 2.0)
        
        if -limit < new_x < limit and -limit < new_z < limit:
            self.x = new_x
            self.z = new_z

    def rotate(self, direction):
        if self.is_attacking: return
        self.angle -= direction * config.ROTATION_SPEED
        self.angle %= 360

    def take_damage(self, amount):
        self.hp -= amount
        if self.hp < 0: self.hp = 0

    def get_hit_target(self, all_pokemons):
        rad = math.radians(self.angle)
        dir_x = math.sin(rad)
        dir_z = math.cos(rad)
        
        # 1. Wall check
        dist_candidates = [config.ATTACK_RANGE]
        limit = config.BOUNDARY
        
        if abs(dir_x) > 1e-6:
            t1 = (limit - self.x) / dir_x
            t2 = (-limit - self.x) / dir_x
            if t1 > 0: dist_candidates.append(t1)
            if t2 > 0: dist_candidates.append(t2)
        if abs(dir_z) > 1e-6:
            t1 = (limit - self.z) / dir_z
            t2 = (-limit - self.z) / dir_z
            if t1 > 0: dist_candidates.append(t1)
            if t2 > 0: dist_candidates.append(t2)
            
        max_wall_dist = min(dist_candidates)
        closest_hit_dist = max_wall_dist
        hit_target = None
        half_width = config.ATTACK_WIDTH / 2.0

        # 2. Entity check
        candidates = []
        for target in all_pokemons:
            if target == self or target.hp <= 0: continue
            
            dx = target.x - self.x
            dz = target.z - self.z
            local_x = dx * math.cos(rad) - dz * math.sin(rad)
            local_z = dx * math.sin(rad) + dz * math.cos(rad)
            
            if 0 < local_z < closest_hit_dist and -half_width < local_x < half_width:
                 candidates.append((local_z, target))

        candidates.sort(key=lambda x: x[0])
        if candidates:
            closest_hit_dist = candidates[0][0]
            hit_target = candidates[0][1]

        return hit_target, closest_hit_dist

    def check_hit(self, all_pokemons):
        target, _ = self.get_hit_target(all_pokemons)
        return target is not None

    def attack(self, all_pokemons):
        self.attack_timer = config.ATTACK_DURATION
        self.is_attacking = True
        
        target, dist = self.get_hit_target(all_pokemons)
        self.actual_beam_length = dist
        
        if target:
            # [CHANGE] Execute Bonus Calculation
            # Calculate ratio BEFORE damage (or current, both work if consistent)
            hp_ratio = target.hp / target.max_hp
            
            target.take_damage(self.attack_power)
            self.damage_dealt_this_step += self.attack_power 
            
            # Formula: 1.0 (at Full HP) -> REWARD_EXECUTE_SCALE (at 0 HP)
            exec_scale = 1.0 + (1.0 - hp_ratio) * (config.REWARD_EXECUTE_SCALE - 1.0)
            self.effective_damage_reward += self.attack_power * exec_scale

            # Backstab Logic
            dx = target.x - self.x
            dz = target.z - self.z
            dist_norm = math.sqrt(dx*dx + dz*dz) + 1e-6
            v_atk = (dx/dist_norm, dz/dist_norm)
            t_fx, t_fz = target.get_forward_vector()
            dot = (v_atk[0] * t_fx) + (v_atk[1] * t_fz)
            
            if dot > 0.2: 
                self.was_backstab_this_step = True

            return True
        return False

    def draw(self):
        glPushMatrix()
        glTranslatef(self.x, self.y, self.z)
        glRotatef(self.angle, 0, 1, 0)
        
        glPushMatrix()
        s = self.model.scale_factor
        glScalef(s, s, s)
        self.model.draw()
        glPopMatrix()

        if self.attack_timer > 0:
            self._draw_beam()
            
        glPopMatrix()

    def _draw_beam(self):
        length = self.actual_beam_length
        radius = config.ATTACK_WIDTH / 2.0
        
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        r, g, b = self.color
        glColor4f(r, g, b, 0.6)

        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluCylinder(quadric, radius, radius, length, 16, 1)
        
        glPushMatrix()
        glTranslatef(0, 0, length)
        gluDisk(quadric, 0, radius, 16, 1)
        glPopMatrix()
        
        gluDisk(quadric, 0, radius, 16, 1)

        gluDeleteQuadric(quadric)
        glPopAttrib()