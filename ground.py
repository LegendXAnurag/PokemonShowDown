# ground.py
from OpenGL.GL import *
import config

def draw_ground():
    """ Draws a solid floor and a wireframe grid """
    half_size = config.BOUNDARY
    step = config.TILE_SIZE
    
    # --- 1. Draw Solid Ground (Grass Color) ---
    glColor3f(0.13, 0.55, 0.13) # Forest Green
    glBegin(GL_QUADS)
    glNormal3f(0, 1, 0)
    glVertex3f(-half_size, -0.01, -half_size) # Slightly below 0 to avoid z-fight with lines
    glVertex3f(half_size, -0.01, -half_size)
    glVertex3f(half_size, -0.01, half_size)
    glVertex3f(-half_size, -0.01, half_size)
    glEnd()

    # --- 2. Draw Grid Lines ---
    glLineWidth(1.0)
    glColor3f(0.2, 0.2, 0.2) # Dark Grey Lines
    
    glBegin(GL_LINES)
    # Lines parallel to X-axis
    z = -half_size
    while z <= half_size:
        glVertex3f(-half_size, 0, z)
        glVertex3f(half_size, 0, z)
        z += step
        
    # Lines parallel to Z-axis
    x = -half_size
    while x <= half_size:
        glVertex3f(x, 0, -half_size)
        glVertex3f(x, 0, half_size)
        x += step
    glEnd()

def draw_walls():
    """ Draws semi-transparent walls at the boundaries """
    h = config.BOUNDARY # half size
    height = 2.0 # Wall height
    
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    # Changed to Amber/Gold Barrier color
    glColor4f(1.0, 0.6, 0.2, 0.3) 

    glBegin(GL_QUADS)
    # Back Wall
    glVertex3f(-h, 0, -h); glVertex3f(h, 0, -h)
    glVertex3f(h, height, -h); glVertex3f(-h, height, -h)
    
    # Front Wall
    glVertex3f(-h, 0, h); glVertex3f(h, 0, h)
    glVertex3f(h, height, h); glVertex3f(-h, height, h)
    
    # Left Wall
    glVertex3f(-h, 0, -h); glVertex3f(-h, 0, h)
    glVertex3f(-h, height, h); glVertex3f(-h, height, -h)
    
    # Right Wall
    glVertex3f(h, 0, -h); glVertex3f(h, 0, h)
    glVertex3f(h, height, h); glVertex3f(h, height, -h)
    glEnd()
    
    glDisable(GL_BLEND)