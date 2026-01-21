"""
Pygame-based visualization for the Pokemon battle environment.
"""

import pygame
import numpy as np
from typing import Dict, Optional
import os


class PokemonVisualizer:
    """
    Visualizer for Pokemon battles using pygame.
    """
    
    def __init__(self, grid_size=10, cell_size=60):
        """
        Initialize the visualizer.
        
        Args:
            grid_size: Size of the grid (grid_size x grid_size)
            cell_size: Size of each cell in pixels
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.window_size = grid_size * cell_size
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size + 200, self.window_size))
        pygame.display.set_caption("Pokemon Battle")
        
        # Colors
        self.COLOR_BACKGROUND = (240, 240, 240)
        self.COLOR_GRID = (200, 200, 200)
        self.COLOR_OBSTACLE = (100, 100, 100)
        self.COLOR_POKEMON = {
            'Fire': (255, 100, 50),
            'Water': (50, 150, 255),
            'Grass': (100, 200, 50),
            'Electric': (255, 255, 50),
            'Ice': (150, 230, 255),
            'Fighting': (200, 100, 50),
            'Psychic': (255, 100, 255),
            'Dragon': (150, 100, 255),
        }
        self.COLOR_HEALTH_BAR = (0, 255, 0)
        self.COLOR_HEALTH_BACKGROUND = (255, 0, 0)
        
        # Font
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)
        
        # Sprite cache
        self.sprite_cache = {}
        
    def load_sprite(self, sprite_path, fallback_color):
        """
        Load a sprite image, or create a colored circle if not available.
        
        Args:
            sprite_path: Path to sprite image
            fallback_color: Color to use if sprite not available
            
        Returns:
            Pygame surface
        """
        if sprite_path in self.sprite_cache:
            return self.sprite_cache[sprite_path]
            
        # Try to load sprite
        if os.path.exists(sprite_path):
            try:
                sprite = pygame.image.load(sprite_path)
                sprite = pygame.transform.scale(sprite, (self.cell_size - 10, self.cell_size - 10))
                self.sprite_cache[sprite_path] = sprite
                return sprite
            except (pygame.error, OSError):
                pass
                
        # Fallback: create colored circle
        surface = pygame.Surface((self.cell_size - 10, self.cell_size - 10), pygame.SRCALPHA)
        pygame.draw.circle(surface, fallback_color, 
                          (self.cell_size // 2 - 5, self.cell_size // 2 - 5), 
                          self.cell_size // 2 - 10)
        self.sprite_cache[sprite_path] = surface
        return surface
        
    def render(self, env, step=0):
        """
        Render the current state of the environment.
        
        Args:
            env: The Pokemon battle environment
            step: Current step number
        """
        self.screen.fill(self.COLOR_BACKGROUND)
        
        # Draw grid
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                  self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.COLOR_GRID, rect, 1)
                
                # Draw obstacles
                if env.grid[y, x] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_OBSTACLE, rect)
                    
        # Draw Pokemon
        for agent, pokemon in env.pokemons.items():
            if not pokemon.is_alive:
                continue
                
            x, y = pokemon.position
            
            # Get sprite or fallback color
            color = self.COLOR_POKEMON.get(pokemon.poke_type, (128, 128, 128))
            sprite = self.load_sprite(pokemon.sprite_path, color)
            
            # Draw sprite
            sprite_rect = sprite.get_rect()
            sprite_rect.center = (x * self.cell_size + self.cell_size // 2,
                                 y * self.cell_size + self.cell_size // 2)
            self.screen.blit(sprite, sprite_rect)
            
            # Draw health bar
            health_bar_width = self.cell_size - 10
            health_bar_height = 5
            health_percentage = pokemon.health / pokemon.base_health
            
            health_bar_x = x * self.cell_size + 5
            health_bar_y = y * self.cell_size + self.cell_size - 10
            
            # Background (red)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BACKGROUND,
                           (health_bar_x, health_bar_y, health_bar_width, health_bar_height))
            # Foreground (green)
            pygame.draw.rect(self.screen, self.COLOR_HEALTH_BAR,
                           (health_bar_x, health_bar_y, 
                            int(health_bar_width * health_percentage), health_bar_height))
                            
        # Draw info panel
        info_x = self.window_size + 10
        info_y = 10
        
        # Draw step counter
        step_text = self.font_large.render(f"Step: {step}", True, (0, 0, 0))
        self.screen.blit(step_text, (info_x, info_y))
        info_y += 40
        
        # Draw level info
        level_text = self.font_medium.render(f"Level: {env.level}", True, (0, 0, 0))
        self.screen.blit(level_text, (info_x, info_y))
        info_y += 30
        
        # Draw Pokemon status
        for i, (agent, pokemon) in enumerate(env.pokemons.items()):
            status = "ALIVE" if pokemon.is_alive else "DEAD"
            color = (0, 150, 0) if pokemon.is_alive else (150, 0, 0)
            
            # Pokemon name
            name_text = self.font_small.render(f"{i}: {pokemon.name}", True, (0, 0, 0))
            self.screen.blit(name_text, (info_x, info_y))
            info_y += 20
            
            # Type
            type_text = self.font_small.render(f"  {pokemon.poke_type}", True, 
                                              self.COLOR_POKEMON.get(pokemon.poke_type, (0, 0, 0)))
            self.screen.blit(type_text, (info_x, info_y))
            info_y += 20
            
            # Health
            health_text = self.font_small.render(f"  HP: {pokemon.health:.0f}/{pokemon.base_health}", 
                                                True, color)
            self.screen.blit(health_text, (info_x, info_y))
            info_y += 25
            
        pygame.display.flip()
        
    def handle_events(self):
        """
        Handle pygame events.
        
        Returns:
            True if should continue, False if should quit
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True
        
    def close(self):
        """Close the visualizer."""
        pygame.quit()
        
    def save_screenshot(self, filename):
        """Save current display as screenshot."""
        pygame.image.save(self.screen, filename)
