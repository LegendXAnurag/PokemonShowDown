# hud.py
# On-screen 2D score overlay using glWindowPos2i + glDrawPixels.
# Works with any OpenGL scene — bypasses the 3D transformation pipeline.

import pygame
from OpenGL.GL import *


class ScoreHUD:
    """Renders a small win-count bar at the top-centre of the screen."""

    TEAM_COLORS = [
        (220,  60,  60),   # Team 0 – red
        ( 60,  60, 220),   # Team 1 – blue
        ( 60, 200,  60),   # Team 2 – green
        (220, 200,  30),   # Team 3 – yellow
    ]

    def __init__(self, font_size=34):
        pygame.font.init()
        self.font   = pygame.font.Font(None, font_size)
        self._raw   = None
        self._w     = 0
        self._h     = 0

    # ------------------------------------------------------------------ #

    def update(self, win_counts: dict):
        """Call whenever win_counts change (or once per frame is fine too)."""
        parts = []
        for team_id, wins in sorted(win_counts.items()):
            parts.append((f"Team {team_id}: {wins} W", team_id))

        # Build a surface wide enough for all teams
        surfaces = []
        sep_surf = self.font.render("   |   ", True, (200, 200, 200))
        for text, team_id in parts:
            color = self.TEAM_COLORS[team_id % len(self.TEAM_COLORS)]
            surfaces.append(self.font.render(text, True, color))

        # Total width
        total_w = sum(s.get_width() for s in surfaces)
        if len(surfaces) > 1:
            total_w += sep_surf.get_width() * (len(surfaces) - 1)
        padding_x, padding_y = 16, 8
        h = surfaces[0].get_height() if surfaces else 30
        full_w = total_w + padding_x * 2
        full_h = h + padding_y * 2

        # Compose onto RGBA surface with semi-transparent dark background
        surf = pygame.Surface((full_w, full_h), pygame.SRCALPHA)
        surf.fill((10, 10, 10, 175))

        x = padding_x
        for i, (text_surf, _) in enumerate(zip(surfaces, parts)):
            surf.blit(text_surf, (x, padding_y))
            x += text_surf.get_width()
            if i < len(surfaces) - 1:
                surf.blit(sep_surf, (x, padding_y))
                x += sep_surf.get_width()

        # Flip vertically: OpenGL expects bottom-row first
        self._raw = pygame.image.tostring(surf, "RGBA", True)
        self._w   = full_w
        self._h   = full_h

    def draw(self, screen_w: int, screen_h: int):
        """Call after all 3D drawing is done, before pygame.display.flip()."""
        if self._raw is None:
            return

        # Place centred, near the top
        x = max(0, (screen_w - self._w) // 2)
        y = screen_h - self._h - 8          # GL y=0 is bottom; top = screen_h

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glWindowPos2i(x, y)
        glDrawPixels(self._w, self._h, GL_RGBA, GL_UNSIGNED_BYTE, self._raw)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
