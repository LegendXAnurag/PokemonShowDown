# utils.py
import numpy as np
import trimesh
import trimesh.transformations as transformations
from OpenGL.GL import *
import config

class ModelLoader:
    def __init__(self, filepath, color=(1, 1, 1), rotation_offset=0):
        """
        :param filepath: Path to .obj/.glb file
        :param color: Fallback color if texture fails (or simple drawing)
        :param rotation_offset: Degrees to rotate around Y-axis (to fix backward models)
        """
        self.vertices = None
        self.faces = None
        self.normals = None
        self.color = color
        self.loaded = False
        self.scale_factor = 1.0
        
        try:
            # Load mesh using trimesh
            scene = trimesh.load(filepath)
            
            # Handle if it loads as a Scene (multiple meshes) or single Geometry
            if isinstance(scene, trimesh.Scene):
                if len(scene.geometry) == 0:
                    raise ValueError("Scene is empty")
                # Dump all geometries into a single mesh
                mesh = trimesh.util.concatenate(
                    tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces) 
                        for g in scene.geometry.values())
                )
            else:
                mesh = scene

            # 1. Center the mesh (Crucial for rotation)
            # This moves the mesh so its center of mass is at (0,0,0)
            mesh.apply_translation(-mesh.centroid)

            # 2. Apply Rotation Correction (Fix Backward models)
            if rotation_offset != 0:
                print(f"Applying rotation correction: {rotation_offset} degrees")
                # Create rotation matrix for Y-axis (0, 1, 0)
                # Convert degrees to radians
                angle_rad = np.radians(rotation_offset)
                matrix = transformations.rotation_matrix(angle_rad, [0, 1, 0])
                mesh.apply_transform(matrix)

            # 3. Calculate Normalization Scale
            # Get the maximum dimension (Length, Width, or Height)
            max_extent = np.max(mesh.extents)
            
            if max_extent > 0:
                self.scale_factor = config.POKEMON_SCALE_SIZE / max_extent
            else:
                self.scale_factor = 1.0

            self.vertices = mesh.vertices
            self.faces = mesh.faces
            self.normals = mesh.vertex_normals
            self.loaded = True
            print(f"Loaded {filepath} | Orig Size: {max_extent:.2f} | Scale Factor: {self.scale_factor:.2f}")

        except Exception as e:
            print(f"Could not load {filepath}, using fallback cube. Error: {e}")
            self.loaded = False
            self.scale_factor = 1.0 # Cube is already 1x1

    def draw(self):
        if self.loaded:
            glColor3f(*self.color)
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)
            
            glVertexPointer(3, GL_DOUBLE, 0, self.vertices)
            glNormalPointer(GL_DOUBLE, 0, self.normals)
            
            glDrawElements(GL_TRIANGLES, len(self.faces) * 3, GL_UNSIGNED_INT, self.faces)
            
            glDisableClientState(GL_VERTEX_ARRAY)
            glDisableClientState(GL_NORMAL_ARRAY)
        else:
            self._draw_cube()

    def _draw_cube(self):
        # Draws a 1x1 cube centered at 0,0,0
        glColor3f(*self.color)
        glBegin(GL_QUADS)
        # Vertices for a 1x1 cube (from -0.5 to 0.5)
        vertices = [
            (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
            (0.5, -0.5, -0.5), (-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5)
        ]
        # Indices for the 6 faces
        faces = [(0,1,2,3), (3,2,7,6), (6,7,5,4), (4,5,1,0), (1,5,7,2), (4,0,3,6)]
        
        for surface in faces:
            for i in surface:
                glVertex3fv(vertices[i])
        glEnd()