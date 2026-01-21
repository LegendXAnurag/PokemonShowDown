import functools
import numpy as np
import pygame
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv

from environment.pokenv import PokemonRL


# Default: Classic trio
# env = PokemonRL()

# Custom team
env = PokemonRL(pokemon_list=["pikachu", "geodude", "pidgey"])

# # Bigger battles
# env = PokemonRL(
#     pokemon_list=["charmander", "bulbasaur", "squirtle", "pikachu", "machop", "onix"],
#     grid_size=15,
#     obs_radius=3
# )

# Create your own Pokemon
from agent.pokemon import Pokemon, PokemonType

# custom_pokemon = Pokemon(
#     name="Mewtwo",
#     poke_type=PokemonType.PSYCHIC,
#     max_hp=150.0,
#     base_attack=18.0,
#     base_defense=0.15,
#     color=(200, 100, 255)
# )

# env = PokemonRL(grid_size=10, obs_radius=2, num_obstacles=5, render_mode="human")
observations, infos = env.reset()

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    env.render()

env.close()


print("hello ")