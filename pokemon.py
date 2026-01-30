# pokemon.py
import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *  # Required for Cylinder
import config
from utils import ModelLoader

class Pokemon:
    def __init__(self, key, data, start_pos):
        self.name = data['name']
        self.hp = data['hp']
        self.max_hp = data['hp']
        self.attack_power = data['attack_power']
        
        # Color needed for the beam
        self.color = data['color_fallback']

        # --- ROTATION FIX START ---
        rot_offset = data.get('rotation_correction', 0)
        self.model = ModelLoader(
            data['model_path'], 
            data['color_fallback'], 
            rotation_offset=rot_offset
        )
        # --- ROTATION FIX END ---
        
        self.x = start_pos[0]
        self.y = (config.POKEMON_SCALE_SIZE / 2.0)
        self.z = start_pos[2]
        self.angle = 0.0 
        
        # Attack Visual State
        self.attack_timer = 0
        self.actual_beam_length = 0.0

    def move_forward(self, speed=None):
        if speed is None:
            speed = config.MOVE_SPEED

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
        self.angle -= direction * config.ROTATION_SPEED
        self.angle %= 360

    def take_damage(self, amount):
        self.hp -= amount
        if self.hp < 0:
            self.hp = 0
        print(f"{self.name} took {amount} damage! HP: {self.hp}/{self.max_hp}")

    def attack(self, all_pokemons):
        """
        Performs an instant beam attack.
        1. Calculates beam length (clipped by walls).
        2. Checks collision with other pokemon.
        3. Sets visual timer.
        """
        self.attack_timer = config.ATTACK_DURATION
        
        # 1. Calculate Beam Direction Vectors
        rad = math.radians(self.angle)
        dir_x = math.sin(rad)
        dir_z = math.cos(rad)
        
        # 2. Calculate Distance to Wall (Raycasting)
        dist_candidates = [config.ATTACK_RANGE] 
        limit = config.BOUNDARY
        
        if dir_x != 0:
            t1 = (limit - self.x) / dir_x
            t2 = (-limit - self.x) / dir_x
            if t1 > 0: dist_candidates.append(t1)
            if t2 > 0: dist_candidates.append(t2)
            
        if dir_z != 0:
            t1 = (limit - self.z) / dir_z
            t2 = (-limit - self.z) / dir_z
            if t1 > 0: dist_candidates.append(t1)
            if t2 > 0: dist_candidates.append(t2)
            
        self.actual_beam_length = min(dist_candidates)
        if self.actual_beam_length > config.ATTACK_RANGE:
            self.actual_beam_length = config.ATTACK_RANGE

        # 3. Check Collisions
        half_width = config.ATTACK_WIDTH / 2.0
        
        for target in all_pokemons:
            if target == self: 
                continue 
            if target.hp <= 0:
                continue 

            dx = target.x - self.x
            dz = target.z - self.z
            
            # Rotate to local space
            local_x = dx * math.cos(rad) - dz * math.sin(rad)
            local_z = dx * math.sin(rad) + dz * math.cos(rad)
            
            # Hit detection (Box approximation matches the visual width)
            if (0 < local_z < self.actual_beam_length) and \
               (-half_width < local_x < half_width):
                   target.take_damage(self.attack_power)

    def draw(self):
        # Draw the model
        glPushMatrix()
        glTranslatef(self.x, self.y, self.z)
        glRotatef(self.angle, 0, 1, 0)
        
        # Save state for the model scaling
        glPushMatrix()
        s = self.model.scale_factor
        glScalef(s, s, s)
        self.model.draw()
        glPopMatrix()

        # Draw Attack Beam if active
        if self.attack_timer > 0:
            self._draw_beam()
            self.attack_timer -= 1
            
        glPopMatrix()

    def _draw_beam(self):
        """ Draws the beam as a cylinder in Local Space """
        length = self.actual_beam_length
        radius = config.ATTACK_WIDTH / 2.0
        
        glPushAttrib(GL_ENABLE_BIT)
        glDisable(GL_LIGHTING) # Make it glow
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE) # Additive blending
        
        r, g, b = self.color
        glColor4f(r, g, b, 0.6)

        # Create a new Quadric for the cylinder
        quadric = gluNewQuadric()
        gluQuadricNormals(quadric, GLU_SMOOTH)
        
        # Draw Cylinder
        # gluCylinder draws along +Z axis from z=0 to z=height
        # (baseRadius, topRadius, height, slices, stacks)
        gluCylinder(quadric, radius, radius, length, 16, 1)
        
        # Draw End Cap (Disk)
        glPushMatrix()
        glTranslatef(0, 0, length)
        # (quad, innerRadius, outerRadius, slices, loops)
        gluDisk(quadric, 0, radius, 16, 1)
        glPopMatrix()
        
        # Draw Start Cap (Disk)
        # Rotate 180 to face outwards from the pokemon body if needed, 
        # but a flat disk at 0 is fine.
        gluDisk(quadric, 0, radius, 16, 1)

        gluDeleteQuadric(quadric)
        glPopAttrib()