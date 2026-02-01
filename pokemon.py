import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
import config
from utils import ModelLoader

class Pokemon:
    def __init__(self, key, data, start_pos):
        self.name = data['name']
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
        
        # [CHANGE] Time-based logic
        self.attack_timer = 0.0  # Float (seconds)
        self.is_attacking = False
        self.actual_beam_length = 0.0

    def update_timers(self, dt):
        """
        [NEW] Updates internal timers based on Delta Time.
        """
        if self.attack_timer > 0:
            self.attack_timer -= dt
            self.is_attacking = True
        else:
            self.attack_timer = 0
            self.is_attacking = False

    def move_forward(self, speed=None):
        # [CHANGE] Prevent movement if attacking
        if self.is_attacking:
            return

        if speed is None: speed = config.MOVE_SPEED
        rad = math.radians(self.angle)
        dx = math.sin(rad) * speed
        dz = math.cos(rad) * speed
        new_x = self.x + dx
        new_z = self.z + dz
        limit = config.BOUNDARY - (config.POKEMON_SCALE_SIZE / 2.0)
        
        # Simple boundary check
        if -limit < new_x < limit and -limit < new_z < limit:
            self.x = new_x
            self.z = new_z

    def rotate(self, direction):
        # [CHANGE] Prevent rotation if attacking
        if self.is_attacking:
            return

        self.angle -= direction * config.ROTATION_SPEED
        self.angle %= 360

    def take_damage(self, amount):
        self.hp -= amount
        if self.hp < 0: self.hp = 0

    def check_hit(self, all_pokemons):
        """
        Checks if an attack WOULD hit. Used for Action Masking.
        """
        rad = math.radians(self.angle)
        dir_x = math.sin(rad)
        dir_z = math.cos(rad)
        
        # Wall Distance
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
            
        beam_length = min(dist_candidates)
        if beam_length > config.ATTACK_RANGE:
            beam_length = config.ATTACK_RANGE

        half_width = config.ATTACK_WIDTH / 2.0
        
        for target in all_pokemons:
            if target == self or target.hp <= 0: 
                continue 

            dx = target.x - self.x
            dz = target.z - self.z
            
            # Local Space
            local_x = dx * math.cos(rad) - dz * math.sin(rad)
            local_z = dx * math.sin(rad) + dz * math.cos(rad)
            
            if (0 < local_z < beam_length) and (-half_width < local_x < half_width):
                return True

        return False

    def attack(self, all_pokemons):
        """
        Sets the attack timer and deals damage.
        """
        # [CHANGE] Set timer in seconds
        self.attack_timer = config.ATTACK_DURATION
        self.is_attacking = True
        
        # Recalculate beam for visuals
        rad = math.radians(self.angle)
        dir_x = math.sin(rad)
        dir_z = math.cos(rad)
        dist_candidates = [config.ATTACK_RANGE]
        limit = config.BOUNDARY
        if abs(dir_x) > 1e-6:
            t1 = (limit - self.x)/dir_x; t2 = (-limit - self.x)/dir_x
            if t1>0: dist_candidates.append(t1)
            if t2>0: dist_candidates.append(t2)
        if abs(dir_z) > 1e-6:
            t1 = (limit - self.z)/dir_z; t2 = (-limit - self.z)/dir_z
            if t1>0: dist_candidates.append(t1)
            if t2>0: dist_candidates.append(t2)
        
        self.actual_beam_length = min(dist_candidates)
        if self.actual_beam_length > config.ATTACK_RANGE:
            self.actual_beam_length = config.ATTACK_RANGE

        hit = self.check_hit(all_pokemons)
        if hit:
            for target in all_pokemons:
                if target != self and target.hp > 0:
                     target.take_damage(self.attack_power)
        return hit

    def draw(self):
        glPushMatrix()
        glTranslatef(self.x, self.y, self.z)
        glRotatef(self.angle, 0, 1, 0)
        
        glPushMatrix()
        s = self.model.scale_factor
        glScalef(s, s, s)
        self.model.draw()
        glPopMatrix()

        # [CHANGE] Check float timer
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