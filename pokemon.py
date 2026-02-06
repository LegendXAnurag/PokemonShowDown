# pokemon.py
import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import config
from utils import ModelLoader

class Pokemon:
    def __init__(self, name, data, start_pos, team_id, team_color):
        self.name = name
        self.hp = data['hp']
        self.max_hp = data['hp']
        self.attack_power = data['attack_power']
        
        # [FIX] Use species color for the model (Original look), store team color for the ring
        self.species_color = data['color_fallback']
        self.team_id = team_id
        self.team_color = team_color
        
        rot_offset = data.get('rotation_correction', 0)
        # Load model with species color, not team color
        self.model = ModelLoader(data['model_path'], self.species_color, rotation_offset=rot_offset)
        
        self.x = start_pos[0]
        self.y = (config.POKEMON_SCALE_SIZE / 2.0)
        self.z = start_pos[2]
        self.angle = 0.0 
        
        self.attack_timer = 0.0 
        self.is_attacking = False
        self.actual_beam_length = 0.0
        
        # Reward tracking
        self.damage_dealt_this_step = 0.0
        self.effective_damage_reward = 0.0
        self.was_backstab_this_step = False 
        self.friendly_fire_damage = 0.0

    def update_timers(self, dt):
        self.damage_dealt_this_step = 0.0 
        self.effective_damage_reward = 0.0
        self.friendly_fire_damage = 0.0
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

    def predict_position(self, direction=1):
        """
        Predicts (x, z) if the agent moves. 
        direction: 1 for forward, -1 for backward (half speed)
        """
        speed = config.MOVE_SPEED if direction == 1 else (config.MOVE_SPEED / 2.0)
        rad = math.radians(self.angle)
        dx = math.sin(rad) * speed * (1 if direction == 1 else -1)
        dz = math.cos(rad) * speed * (1 if direction == 1 else -1)
        return self.x + dx, self.z + dz

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
        initial_hp=self.hp
        self.hp -= amount
        if self.hp<0 and initial_hp>0:
            pass
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
            if target.team_id == self.team_id:
                target.take_damage(self.attack_power)
                self.friendly_fire_damage += self.attack_power
                return True 

            hp_ratio = target.hp / target.max_hp
            target.take_damage(self.attack_power)
            self.damage_dealt_this_step += self.attack_power 
            
            # exec_scale = 1.0 + (1.0 - hp_ratio) * (config.REWARD_EXECUTE_SCALE - 1.0)
            # self.effective_damage_reward += self.attack_power * exec_scale
            self.effective_damage_reward = 1.0 + (1.0 - hp_ratio) * (config.REWARD_EXECUTE_SCALE - 1.0)

            dx = target.x - self.x
            dz = target.z - self.z
            dist_norm = math.sqrt(dx*dx + dz*dz) + 1e-6
            v_atk = (dx/dist_norm, dz/dist_norm)
            t_fx, t_fz = target.get_forward_vector()
            dot = (v_atk[0] * t_fx) + (v_atk[1] * t_fz)
            
            if dot > -0.5: 
                self.was_backstab_this_step = True

            return True
        return False

    def draw(self):
        # [FIX] Draw Team Indicator Ring on the ground
        self.draw_team_indicator()

        glPushMatrix()
        glTranslatef(self.x, self.y, self.z)
        glRotatef(self.angle, 0, 1, 0)
        
        glPushMatrix()
        s = self.model.scale_factor
        glScalef(s, s, s)
        
        # [FIX] Reset color to white so texture/material colors show correctly
        glColor3f(1, 1, 1) 
        self.model.draw()
        glPopMatrix()

        if self.attack_timer > 0:
            self._draw_beam()
            
        glPopMatrix()

    def draw_team_indicator(self):
        """Draws a colored ring below the pokemon to indicate team."""
        glPushMatrix()
        glTranslatef(self.x, 0.05, self.z) # Slightly above ground
        
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        r, g, b = self.team_color
        glColor4f(r, g, b, 0.7) # Slightly transparent
        
        quadric = gluNewQuadric()
        # Inner radius 0.4, Outer radius 0.6
        gluDisk(quadric, 0.4, 0.7, 20, 1)
        gluDeleteQuadric(quadric)
        
        glDisable(GL_BLEND)
        glEnable(GL_LIGHTING)
        
        glPopMatrix()
    # In pokemon.py

    def draw_model(self):
        """Draws the opaque parts: Team Ring and Pokemon Body."""
        # Draw Team Indicator Ring (Opaque/Alpha Test)
        self.draw_team_indicator()

        glPushMatrix()
        glTranslatef(self.x, self.y, self.z)
        glRotatef(self.angle, 0, 1, 0)
        
        glPushMatrix()
        s = self.model.scale_factor
        glScalef(s, s, s)
        
        # Reset color to white so texture/material colors show correctly
        glColor3f(1, 1, 1) 
        self.model.draw()
        glPopMatrix()
            
        glPopMatrix()

    def draw_beam(self):
        """Draws the transparent attack beam."""
        if self.attack_timer <= 0:
            return

        glPushMatrix()
        # Apply the same transformations so the beam starts from the Pokemon
        glTranslatef(self.x, self.y, self.z)
        glRotatef(self.angle, 0, 1, 0)

        length = self.actual_beam_length
        radius = config.ATTACK_WIDTH / 2.0
        
        glPushAttrib(GL_ENABLE_BIT | GL_DEPTH_BUFFER_BIT) # Save Depth buffer bit too
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        # [FIX] Disable Depth Mask. 
        # The beam is checked against walls (Depth Test), 
        # but it won't write to the depth buffer, so it won't hide objects drawn after it.
        glDepthMask(GL_FALSE) 
        
        r, g, b = self.species_color
        glColor4f(r, g, b, 0.4)

        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        gluCylinder(quadric, radius, radius, length, 16, 1)
        
        glPushMatrix()
        glTranslatef(0, 0, length)
        gluDisk(quadric, 0, radius, 16, 1)
        glPopMatrix()
        
        gluDisk(quadric, 0, radius, 16, 1)

        gluDeleteQuadric(quadric)
        
        # [FIX] Restore attributes (including Depth Mask = True)
        glPopAttrib() 
        glPopMatrix()
    def _draw_beam(self):
        length = self.actual_beam_length
        radius = config.ATTACK_WIDTH / 2.0
        
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        
        r, g, b = self.species_color
        glColor4f(r, g, b, 0.4)

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