# main.py
import math  # Needed for atan2
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

import config
import pokemon_data
from ground import draw_ground, draw_walls
from pokemon import Pokemon

def init_gl():
    # Sky Blue Color
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

def draw_hp_bar(pokemon, camera_pos):
    """ 
    Draws a small HP bar above the pokemon.
    Uses Billboarding (rotation) to ensure it always faces the camera.
    """
    glPushMatrix()
    glTranslatef(pokemon.x, pokemon.y + 2.0, pokemon.z)
    
    # --- BILLBOARDING LOGIC START ---
    # Calculate vector from Pokemon to Camera
    cam_x, cam_y, cam_z = camera_pos[0], camera_pos[1], camera_pos[2]
    dx = cam_x - pokemon.x
    dz = cam_z - pokemon.z
    
    # Calculate angle to face the camera
    # math.atan2(x, y) in this context (dx, dz) gives the angle relative to Z-axis
    angle_radians = math.atan2(dx, dz)
    angle_degrees = math.degrees(angle_radians)
    
    # Apply rotation around Y-axis so the bar faces the camera
    glRotatef(angle_degrees, 0, 1, 0)
    # --- BILLBOARDING LOGIC END ---

    width = 1.0
    height = 0.15
    hp_ratio = pokemon.hp / pokemon.max_hp
    
    glDisable(GL_LIGHTING)
    
    # Background (Red)
    glColor3f(1, 0, 0)
    glBegin(GL_QUADS)
    # Note: Drawn along X-axis, facing Z. The rotation above aligns Z to camera.
    glVertex3f(-width/2, 0, 0); glVertex3f(width/2, 0, 0)
    glVertex3f(width/2, height, 0); glVertex3f(-width/2, height, 0)
    glEnd()
    
    # Foreground (Green)
    z_offset = 0.05 # Prevent Z-fighting
    
    if hp_ratio > 0:
        glColor3f(0, 1, 0)
        current_width = width * hp_ratio
        glBegin(GL_QUADS)
        glVertex3f(-width/2, 0, z_offset); glVertex3f(-width/2 + current_width, 0, z_offset)
        glVertex3f(-width/2 + current_width, height, z_offset); glVertex3f(-width/2, height, z_offset)
        glEnd()

    glEnable(GL_LIGHTING)
    glPopMatrix()

def main():
    pygame.init()
    pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), DOUBLEBUF | OPENGL)
    
    base_title = config.WINDOW_TITLE
    controls_text = " | [WASD] Move, [SPACE] Attack, [TAB] Switch Poke, [1-4] View"
    pygame.display.set_caption(base_title + controls_text)
    
    init_gl()
    
    # Initialize Pokemon
    p1 = Pokemon('pikachu', pokemon_data.POKEMON_DB['pikachu'], (-2, 0, 0))
    p2 = Pokemon('charmander', pokemon_data.POKEMON_DB['charmander'], (2, 0, 0))
    
    pokemons = [p1, p2]
    active_idx = 0
    view_idx = 0
    
    clock = pygame.time.Clock()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                # --- ACTION CONTROLS ---
                if event.key == pygame.K_SPACE:
                    if pokemons[active_idx].hp > 0:
                        pokemons[active_idx].attack(pokemons)
                        print(f"{pokemons[active_idx].name} used Attack!")

                # --- POKEMON SELECTION ---
                if event.key == pygame.K_TAB:
                    active_idx = (active_idx + 1) % len(pokemons)
                if event.key == pygame.K_F1: active_idx = 0
                if event.key == pygame.K_F2 and len(pokemons) > 1: active_idx = 1

                # --- VIEW SELECTION ---
                if event.key == pygame.K_v:
                    view_idx = (view_idx + 1) % len(config.VIEWS)
                if event.key == pygame.K_1: view_idx = 0
                if event.key == pygame.K_2: view_idx = 1
                if event.key == pygame.K_3: view_idx = 2
                if event.key == pygame.K_4: view_idx = 3

        # Input & Logic
        keys = pygame.key.get_pressed()
        active_pokemon = pokemons[active_idx]
        
        if active_pokemon.hp > 0:
            if keys[pygame.K_w]: active_pokemon.move_forward()
            if keys[pygame.K_s]: active_pokemon.move_forward(speed=-0.05) 
            if keys[pygame.K_a]: active_pokemon.rotate(-1)
            if keys[pygame.K_d]: active_pokemon.rotate(1)

        # Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Get current camera settings
        cam = config.VIEWS[view_idx]
        gluLookAt(cam[0], cam[1], cam[2], 0, 1, 0, 0, 1, 0)
        
        draw_ground()
        draw_walls()
        
        for p in pokemons:
            if p.hp > 0:
                p.draw()
                # Pass the camera position (cam) to the draw function
                draw_hp_bar(p, cam)

        # Active Indicator
        if active_pokemon.hp > 0:
            glPushMatrix()
            glTranslatef(active_pokemon.x, active_pokemon.y + 1.5, active_pokemon.z)
            if active_idx == 0: glColor3f(0, 1, 0)
            else: glColor3f(1, 0, 0)
            glScalef(0.2, 0.2, 0.2)
            
            # Simple rotation for the indicator too, 
            # or keep it 3D spinning. Let's keep it spinning or static 3D.
            # But let's apply the same logic if we want it to face us.
            # For now, 3D pyramid shape is fine from all angles.
            glBegin(GL_TRIANGLES)
            glVertex3f(-0.5, 0.0, 0); glVertex3f(0.5, 0.0, 0); glVertex3f(0, 1.0, 0)
            glVertex3f(-0.5, 0.0, 0); glVertex3f(0.5, 0.0, 0); glVertex3f(0, -0.5, 0)
            glEnd()
            glPopMatrix()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()