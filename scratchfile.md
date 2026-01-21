# MINIMALISTIC RL CoLearning
Multiple agents influence each others learning
Task : To develop an excellent co learning algorithm and apply it on this game of PokemonShowDown

Pokemon attributes 
pokeType
ActionSpace( 4 directions to move, attack etc) (to dodge attacks, you can move in other direction to attack)

pokemon size 1x1
Pokemon attack is like a beam, damages opponent if it is in the line of attack
Base attack and health for all pokemons is same for Minimalistic

DamageMultiplier advantage/disadvantage based on pokemon type
hence pokemon with disadvantage should try to dodge more as its health will reduce fast

entropy_coefficient should be decent so that the model explores all of its action space instead of just attacking

## Environment
Training level 1 
5x5 grid, no obstacles (nothing to dodge attacks), 1v1 pokemon ( pick any 2 pokemons randomly out of the available list)
Training level 2
6x6 grid, no obstacles (nothing to dodge attacks),  1v1 pokemon ( pick any 2 pokemons randomly out of the available list)
Training level 3
6x6 grid, 2-3 obtacles (obstacles should not share a edge/vertex),  1v1 pokemon ( pick any 2 pokemons randomly out of the available list)
Training level 4 
10x10 grid, 4-5 obstacles( obstacles should not share a edge/vertex), 1v1v1 pokemon( pick any 3 pokemon randomly out of available list)
Training level 5
10x10 grid, 4-5 obstacles( obstacles should not share a edge/vertex), 1v1v1v1v1 pokemon( pick any 5 pokemon randomly out of available list)
Training level 6
10x10 grid, 4-5 obstacles( obstacles should not share a edge/vertex), 2v2 pokemon( pick any 4 pokemon randomly out of available list)
in this level, pokemon will team up, they will learn not to attack their teammates rather attack opponents

### Rewards for
defeating opponent
damaging opponent 
wining condition

### Penalty for
taking damage
getting defeated
time elasped (so that pokemons have to fight)


## Code Structure
A general class for pokemon
A class for the Environment
Current game visualizer
Model training on each level
A static file for listing the pokemon and its types (atleast 7+) and then containing reference for the pokemon sprite
a static folder containing pokemon sprite
visualization after each fixed amount of steps
visualization after each level

For agent training we can use dedicated algorithms suxh as MAPPO or IPPO. For agent architecture and training loop we use torchrl. 
For environment simulation we use pettingzoo and for the rendering we use pygame. 

```python
from pettingzoo import ParallelEnv
class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self):
        pass

    def reset(self, seed=None, options=None):
        pass

    def step(self, actions):
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
```
pettingzoo environment format