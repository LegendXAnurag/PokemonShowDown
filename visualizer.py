"""
Pygame-based visualizer for Pokemon Showdown with beam attacks
"""
import pygame
import numpy as np
from config import RENDER_CONFIG

class Beam:
    """Represents a moving beam attack"""
    def __init__(self, start_x, start_y, direction, color, attacker_id):
        self.x = start_x
        self.y = start_y
        self.direction = direction  # (dx, dy)
        self.color = color
        self.attacker_id = attacker_id
        self.active = True
        self.speed = 0.3  # cells per frame
        self.progress = 0.0  # sub-cell position
        
    def update(self):
        """Update beam position"""
        if not self.active:
            return
        
        self.progress += self.speed
        
        # Move to next cell when progress >= 1
        while self.progress >= 1.0:
            self.progress -= 1.0
            self.x += self.direction[0]
            self.y += self.direction[1]
    
    def get_render_pos(self):
        """Get interpolated position for rendering"""
        return (
            self.x + self.direction[0] * self.progress,
            self.y + self.direction[1] * self.progress
        )


class PokemonRenderer:
    def __init__(self, grid_size):
        """
        Initialize the renderer
        
        Args:
            grid_size: Size of the game grid
        """
        self.grid_size = grid_size
        self.cell_size = RENDER_CONFIG['cell_size']
        self.fps = RENDER_CONFIG['fps']
        self.colors = RENDER_CONFIG['colors']
        
        self.screen_width = grid_size * self.cell_size
        self.screen_height = grid_size * self.cell_size + 100  # Extra space for info
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Pokemon Showdown")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Track active beams
        self.beams = []
        
    def add_beam(self, start_x, start_y, direction, color, attacker_id):
        """Add a new beam attack"""
        beam = Beam(start_x, start_y, direction, color, attacker_id)
        self.beams.append(beam)
    
    def update_beams(self, obstacles, pokemons):
        """Update all beams and check for collisions"""
        beams_to_remove = []
        hits = []
        
        for beam in self.beams:
            if not beam.active:
                beams_to_remove.append(beam)
                continue
            
            beam.update()
            
            # Check if beam is out of bounds
            if (beam.x < 0 or beam.x >= self.grid_size or 
                beam.y < 0 or beam.y >= self.grid_size):
                beam.active = False
                beams_to_remove.append(beam)
                continue
            
            # Check if beam hit obstacle
            if (beam.x, beam.y) in obstacles:
                beam.active = False
                beams_to_remove.append(beam)
                continue
            
            # Check if beam hit a pokemon
            for agent, pokemon in pokemons.items():
                if (pokemon.alive and 
                    pokemon.x == beam.x and 
                    pokemon.y == beam.y and
                    agent != beam.attacker_id):
                    beam.active = False
                    beams_to_remove.append(beam)
                    hits.append((agent, beam.attacker_id))
                    break
        
        # Remove inactive beams
        for beam in beams_to_remove:
            if beam in self.beams:
                self.beams.remove(beam)
        
        return hits
        
    def render(self, pokemons, obstacles, grid_size, mode='human'):
        """
        Render the current state
        
        Args:
            pokemons: Dictionary of agent -> Pokemon
            obstacles: Set of obstacle positions
            grid_size: Size of grid
            mode: Render mode ('human' or 'rgb_array')
        """
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Draw grid
        for x in range(grid_size + 1):
            pygame.draw.line(
                self.screen,
                self.colors['grid'],
                (x * self.cell_size, 0),
                (x * self.cell_size, grid_size * self.cell_size),
                1
            )
        
        for y in range(grid_size + 1):
            pygame.draw.line(
                self.screen,
                self.colors['grid'],
                (0, y * self.cell_size),
                (grid_size * self.cell_size, y * self.cell_size),
                1
            )
        
        # Draw obstacles
        for ox, oy in obstacles:
            rect = pygame.Rect(
                ox * self.cell_size + 2,
                oy * self.cell_size + 2,
                self.cell_size - 4,
                self.cell_size - 4
            )
            pygame.draw.rect(self.screen, self.colors['obstacle'], rect)
        
        # Update and draw beams
        self.update_beams(obstacles, pokemons)
        for beam in self.beams:
            if beam.active:
                render_x, render_y = beam.get_render_pos()
                center = (
                    int(render_x * self.cell_size + self.cell_size // 2),
                    int(render_y * self.cell_size + self.cell_size // 2)
                )
                
                # Draw beam as a glowing circle
                pygame.draw.circle(self.screen, beam.color, center, self.cell_size // 4)
                # Add glow effect
                glow_color = tuple(min(255, c + 50) for c in beam.color)
                pygame.draw.circle(self.screen, glow_color, center, self.cell_size // 4, 2)
        
        # Draw pokemons
        for agent, pokemon in pokemons.items():
            if not pokemon.alive:
                continue
            
            x = pokemon.x * self.cell_size
            y = pokemon.y * self.cell_size
            
            # Draw pokemon circle
            center = (x + self.cell_size // 2, y + self.cell_size // 2)
            radius = self.cell_size // 3
            
            pygame.draw.circle(self.screen, pokemon.color, center, radius)
            pygame.draw.circle(self.screen, (0, 0, 0), center, radius, 2)
            
            # Draw health bar
            bar_width = self.cell_size - 10
            bar_height = 6
            bar_x = x + 5
            bar_y = y + self.cell_size - 15
            
            # Background (red)
            pygame.draw.rect(
                self.screen,
                (200, 0, 0),
                (bar_x, bar_y, bar_width, bar_height)
            )
            
            # Health (green)
            health_width = int(bar_width * pokemon.get_health_ratio())
            pygame.draw.rect(
                self.screen,
                (0, 200, 0),
                (bar_x, bar_y, health_width, bar_height)
            )
            
            # Draw type indicator
            type_text = self.small_font.render(pokemon.type[:3].upper(), True, (0, 0, 0))
            text_rect = type_text.get_rect(center=(x + self.cell_size // 2, y + 10))
            self.screen.blit(type_text, text_rect)
            
            # Draw cooldown indicator if on cooldown
            if pokemon.attack_cooldown > 0:
                cooldown_text = self.small_font.render(str(pokemon.attack_cooldown), True, (255, 0, 0))
                cooldown_rect = cooldown_text.get_rect(center=(x + self.cell_size - 10, y + 10))
                # Draw small circle background
                pygame.draw.circle(self.screen, (255, 255, 255), cooldown_rect.center, 8)
                pygame.draw.circle(self.screen, (255, 0, 0), cooldown_rect.center, 8, 1)
                self.screen.blit(cooldown_text, cooldown_rect)
        
        # Draw info panel
        info_y = grid_size * self.cell_size + 10
        
        alive_count = sum(1 for p in pokemons.values() if p.alive)
        info_text = self.font.render(f"Alive: {alive_count}/{len(pokemons)}", True, (0, 0, 0))
        self.screen.blit(info_text, (10, info_y))
        
        # Show pokemon details
        details_x = 10
        details_y = info_y + 30
        
        for i, (agent, pokemon) in enumerate(pokemons.items()):
            if not pokemon.alive:
                status = "DEAD"
                color = (150, 150, 150)
            else:
                status = f"HP: {int(pokemon.health)}"
                color = (0, 0, 0)
            
            detail_text = self.small_font.render(
                f"{pokemon.name}: {status}",
                True,
                color
            )
            self.screen.blit(detail_text, (details_x + (i % 2) * 250, details_y + (i // 2) * 25))
        
        if mode == 'human':
            pygame.display.flip()
            self.clock.tick(self.fps)
        elif mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                axes=(1, 0, 2)
            )
    
    def close(self):
        """Clean up pygame resources"""
        pygame.quit()