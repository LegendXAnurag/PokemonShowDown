# testbench.py
# [FIX BUG-06] Removed config.VIEWS (never existed), fixed Pokemon() constructor.
# Minimal 2-Pokemon sandbox — no AI, no teams, just manual WASD control.
import math
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

import config
import pokemon_data
from ground import draw_ground, draw_walls
from pokemon import Pokemon

# Static camera positions to cycle through
VIEWS = [
    (0, 15, 15),   # Default isometric
    (0, 25, 0),    # Top-down
    (0, 8, 10),    # Low angle
    (15, 10, 0),   # Side view
]

def init_gl():
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
        glColor4f(0, 1, 0, 0.9)
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

    controls_text = "Pokemon 3D Testbench | [WASD] Move, [SPACE] Attack, [TAB] Switch, [V] View"
    pygame.display.set_caption(controls_text)

    init_gl()

    # [FIX BUG-06] Pokemon() requires: name, data, pos, team_id, team_color
    red_color = config.TEAMS_SETUP[0]["color"]
    blue_color = config.TEAMS_SETUP[1]["color"]
    p1 = Pokemon('pikachu',    pokemon_data.POKEMON_DB['pikachu'],    (-2, 0, 0), 0, red_color)
    p2 = Pokemon('charmander', pokemon_data.POKEMON_DB['charmander'], ( 2, 0, 0), 1, blue_color)

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
                if event.key == pygame.K_SPACE:
                    if pokemons[active_idx].hp > 0:
                        pokemons[active_idx].attack(pokemons)
                        print(f"{pokemons[active_idx].name} used Attack!")

                if event.key == pygame.K_TAB:
                    active_idx = (active_idx + 1) % len(pokemons)
                if event.key == pygame.K_F1: active_idx = 0
                if event.key == pygame.K_F2 and len(pokemons) > 1: active_idx = 1

                if event.key == pygame.K_v:
                    view_idx = (view_idx + 1) % len(VIEWS)
                if event.key == pygame.K_1: view_idx = 0
                if event.key == pygame.K_2: view_idx = 1
                if event.key == pygame.K_3: view_idx = 2
                if event.key == pygame.K_4: view_idx = 3

        # Update timers
        for p in pokemons:
            p.update_timers(config.DT)

        keys = pygame.key.get_pressed()
        active_pokemon = pokemons[active_idx]

        if active_pokemon.hp > 0:
            if keys[pygame.K_w]: active_pokemon.move_forward()
            if keys[pygame.K_s]: active_pokemon.move_forward(speed=-config.MOVE_SPEED / 2)
            if keys[pygame.K_a]: active_pokemon.rotate(-1)
            if keys[pygame.K_d]: active_pokemon.rotate(1)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        cam = VIEWS[view_idx]
        gluLookAt(cam[0], cam[1], cam[2], 0, 0, 0, 0, 1, 0)

        draw_ground()
        draw_walls()

        for p in pokemons:
            if p.hp > 0:
                p.draw()
                draw_hp_bar(p, cam)

        # Active indicator
        if active_pokemon.hp > 0:
            glPushMatrix()
            glTranslatef(active_pokemon.x, active_pokemon.y + 1.5, active_pokemon.z)
            glDisable(GL_LIGHTING)
            glColor3f(0, 1, 0) if active_idx == 0 else glColor3f(1, 0, 0)
            glScalef(0.2, 0.2, 0.2)
            glBegin(GL_TRIANGLES)
            glVertex3f(-0.5, 0.0, 0); glVertex3f(0.5, 0.0, 0); glVertex3f(0, 1.0, 0)
            glEnd()
            glEnable(GL_LIGHTING)
            glPopMatrix()

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()