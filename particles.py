# particles.py
# FEATURE-07: Visual-only particle attack effects.
# IMPORTANT: Hit detection is computed in pokemon.py attack() BEFORE this is called.
# Particles are spawned along actual_beam_length — the SAME value used to detect
# the hit — so the visual always matches what registered as a hit.

import math
import random
from OpenGL.GL import *

# Per-species particle colors (RGB float)
_SPECIES_COLORS = {
    "pikachu":    (1.0, 0.95, 0.0),   # Electric yellow
    "charmander": (1.0, 0.42, 0.0),   # Fire orange
    "squirtle":   (0.1, 0.65, 1.0),   # Water blue
    "bulbasaur":  (0.2, 0.95, 0.2),   # Grass green
    "rhyhorn":    (0.72, 0.68, 0.60), # Rock grey-brown
    "eevee":      (0.85, 0.55, 0.22), # Normal brown
}
_DEFAULT_COLOR = (1.0, 1.0, 1.0)


class _Particle:
    """Single particle: position, velocity, color, lifetime."""
    __slots__ = ("x", "y", "z", "vx", "vy", "vz", "color", "lifetime", "max_lifetime", "size", "alive")

    def __init__(self, x, y, z, vx, vy, vz, color, lifetime, size=0.06):
        self.x, self.y, self.z = x, y, z
        self.vx, self.vy, self.vz = vx, vy, vz
        self.color = color
        self.lifetime = lifetime
        self.max_lifetime = max(lifetime, 1e-6)
        self.size = size
        self.alive = True

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt
        self.vy -= 3.5 * dt   # gentle gravity pull-down
        self.lifetime -= dt
        if self.lifetime <= 0:
            self.alive = False


class ParticleSystem:
    """Manages all live particles and their rendering."""

    def __init__(self):
        self.particles = []  # list of _Particle

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def emit_attack(self, pokemon, beam_length: float, hit_target: bool):
        """
        Spawn particles when `pokemon` fires an attack.

        :param pokemon:      The attacking Pokemon instance (provides angle, pos, name)
        :param beam_length:  actual_beam_length — the exact hit/wall distance already
                             computed by pokemon.attack().  Particles are spawned along
                             this range so visual always matches hit registration.
        :param hit_target:   True if an *enemy* was hit (triggers burst effect).
        """
        if beam_length <= 0:
            return

        rad = math.radians(pokemon.angle)
        dir_x = math.sin(rad)
        dir_z = math.cos(rad)
        perp_x = -dir_z         # perpendicular (left-right of beam)
        perp_z =  dir_x

        color = _SPECIES_COLORS.get(pokemon.name.lower(), _DEFAULT_COLOR)
        base_y = pokemon.y + 0.3  # slightly above ground level

        # ---- Beam trail ------------------------------------------------
        num_trail = max(4, int(beam_length * 10))
        for _ in range(num_trail):
            t = random.uniform(0.05, beam_length)
            scatter = random.uniform(-0.12, 0.12)

            px = pokemon.x + dir_x * t + perp_x * scatter
            py = base_y + random.uniform(-0.08, 0.2)
            pz = pokemon.z + dir_z * t + perp_z * scatter

            vx = perp_x * random.uniform(-0.6, 0.6) + dir_x * random.uniform(0.2, 1.2)
            vy = random.uniform(0.3, 1.8)
            vz = perp_z * random.uniform(-0.6, 0.6) + dir_z * random.uniform(0.2, 1.2)

            self.particles.append(
                _Particle(px, py, pz, vx, vy, vz, color, random.uniform(0.25, 0.55))
            )

        # ---- Hit burst at beam tip (only for enemy hits) ---------------
        if hit_target:
            tip_x = pokemon.x + dir_x * beam_length
            tip_y = base_y
            tip_z = pokemon.z + dir_z * beam_length

            for _ in range(18):
                angle = random.uniform(0, 2 * math.pi)
                speed = random.uniform(1.2, 3.5)
                vx = math.cos(angle) * speed
                vy = random.uniform(1.5, 4.5)
                vz = math.sin(angle) * speed
                self.particles.append(
                    _Particle(tip_x, tip_y, tip_z, vx, vy, vz, color,
                              random.uniform(0.35, 0.75), size=0.09)
                )

    def update(self, dt: float):
        """Advance all particles; remove dead ones."""
        self.particles = [p for p in self.particles if p.alive]
        for p in self.particles:
            p.update(dt)

    def draw(self):
        """Render all live particles using additive blending (glow effect)."""
        if not self.particles:
            return

        glPushAttrib(GL_ENABLE_BIT | GL_POINT_BIT | GL_CURRENT_BIT)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)    # additive — no dark halos
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        glDepthMask(GL_FALSE)               # don't write to depth so particles don't occlude models

        glPointSize(5.5)
        glBegin(GL_POINTS)
        for p in self.particles:
            alpha = (p.lifetime / p.max_lifetime) * 0.92
            r, g, b = p.color
            glColor4f(r, g, b, alpha)
            glVertex3f(p.x, p.y, p.z)
        glEnd()

        glDepthMask(GL_TRUE)
        glPopAttrib()
