# Pokemon 3D Team Battle Engine

A **3D multi-agent reinforcement learning** battle engine built with Python, PyTorch, and OpenGL. Pokémon agents are trained using **MAPPO (Multi-Agent Proximal Policy Optimization)** to fight in a top-down 3D arena. A pre-trained AI can then be watched battling itself or challenged directly by a human player.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
  - [The RL Environment](#the-rl-environment)
  - [Observation Space](#observation-space)
  - [Action Space](#action-space)
  - [Reward System](#reward-system)
  - [Neural Network Model](#neural-network-model)
- [Pokémon Roster](#pokémon-roster)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Project](#running-the-project)
- [Controls](#controls)
- [Configuration](#configuration)
- [Training](#training)
- [File Reference](#file-reference)

---

## Overview

This project is a complete pipeline for training and visualizing autonomous Pokémon battle agents using deep reinforcement learning. The environment is a bounded 10×10 world-unit 3D arena where two teams of Pokémon (default: 2v2) fight to eliminate the opposing team.

The system has three main modes:
1. **Training** — Run MAPPO to train agents from scratch (or resume from a checkpoint)
2. **AI vs AI Viewer** — Watch the trained model control all agents autonomously
3. **Human vs AI** — You play as a Pokémon of your choice against the trained AI

---

## Features

- **MAPPO Training** — Parameter-sharing PPO across all agents in both teams, with GAE advantage estimation
- **Action Masking** — Invalid actions (e.g., moving into a wall or attacking when no target is in range) are masked before sampling, ensuring the agent never wastes actions
- **LIDAR Observation** — Each agent perceives its surroundings via 16 rays cast in 360°, each returning 6 channels of information (distance, is-wall, is-enemy, is-teammate, target HP, target facing)
- **Backstab Mechanic** — Attacking a target from behind or the side awards a bonus reward multiplier, incentivizing tactical flanking behaviour
- **Friendly Fire Penalty** — Hitting a teammate does zero HP damage but incurs a massive reward penalty, preventing agents from harming their own team
- **3D OpenGL Rendering** — Real `.glb` 3D models rendered via `trimesh` + OpenGL with auto-scaling, rotation correction, and lighting
- **Particle Effects** — Per-species colored particle bursts on attack (beam trail + hit explosion), rendered with additive blending for a glow effect
- **On-Screen HUD** — Team win-counter overlay rendered as a 2D OpenGL pixel draw, visible at all times during the battle
- **Multiple Camera Modes** — Overhead, Follow (human mode only), and auto-rotating Orbit cameras
- **HP Bars** — Billboarded health bars floating above every Pokémon, always facing the camera
- **Human Player Mode** — Mouse-aimed rotation, WASD movement, Space/click attack, with a species-selection screen before the battle
- **Configurable Teams** — Override team composition at runtime via `--teams 2v2` / `--teams 1v3` etc.
- **CSV Training Logs** — Every training update is logged (reward, value loss, total loss, entropy, FPS) to a `.csv` file in `logs/`
- **Checkpointing** — Model + optimizer state saved periodically; training can be resumed from any checkpoint

---

## Project Structure

```
PokemonShowDown/
│
├── train.py            # Main training script (MAPPO)
├── test.py             # AI vs AI battle viewer
├── test_human.py       # Human vs AI battle mode
├── testbench.py        # Minimal 2-Pokémon sandbox (no AI, manual control)
├── verify_dist.py      # Spawn-distance constraint unit test
│
├── config.py           # All hyperparameters and simulation settings
├── custom_env.py       # PettingZoo ParallelEnv — the core RL environment
├── model.py            # Neural network architecture (Actor-Critic)
├── pokemon.py          # Pokemon class — movement, attack, rendering
├── pokemon_data.py     # Species database (HP, attack, model path, colors)
├── ground.py           # OpenGL ground plane and arena wall rendering
├── hud.py              # 2D score overlay (glWindowPos2i + glDrawPixels)
├── particles.py        # Particle system for attack visual effects
├── utils.py            # ModelLoader — loads .glb files via trimesh → OpenGL
│
├── assets/             # .glb 3D model files for each Pokémon species
│   ├── pikachu.glb
│   ├── charmander.glb
│   ├── squirtle.glb
│   ├── bulbasaur.glb
│   ├── rhyhorn.glb
│   └── eevee.glb
│
├── checkpoints/        # Saved model weights (.pt files)
│   └── pokemon_team_battle_2v2_backstab_feb6.pt
│
└── logs/               # CSV training logs (auto-created during training)
```

---

## Architecture

### The RL Environment

`custom_env.py` implements `PokemonBattleEnv`, a **PettingZoo `ParallelEnv`**. All agents step simultaneously each frame.

**Episode flow:**
1. `reset()` — Agents are assigned random species and spawned in team clusters at opposite sides of the arena. Positions respect minimum teammate and inter-enemy distance constraints.
2. `step(actions)` — Actions are applied (move forward/back, rotate left/right, attack, no-op). Rewards are computed based on damage dealt, damage taken, backstab bonus, friendly fire penalty, survival, win/loss, and timeout.
3. The episode ends when only one team has living Pokémon, or after `MAX_STEPS_PER_EPISODE` (4000) steps (timeout = large negative reward).

**Vectorization for training**: The environment is wrapped by SuperSuit (`pettingzoo_env_to_vec_env_v1` + `concat_vec_envs_v1`) to convert the multi-agent parallel env into a standard vectorized Gym-compatible env for the PPO loop.

---

### Observation Space

Each agent receives a **98-dimensional float32 vector**:

| Component | Size | Description |
|-----------|------|-------------|
| Self state | 2 | Normalized HP (`hp / max_hp`), normalized attack cooldown |
| LIDAR rays | 16 × 6 = 96 | 16 rays cast at equal angular intervals around the agent |

**Each LIDAR ray has 6 channels:**

| Channel | Description |
|---------|-------------|
| `dist` | Normalized distance to the first obstacle (0 = at the obstacle, 1 = max range) |
| `is_wall` | 1.0 if the ray hit a boundary wall, else 0.0 |
| `is_enemy` | 1.0 if the ray hit an enemy Pokémon, else 0.0 |
| `is_teammate` | 1.0 if the ray hit a teammate, else 0.0 |
| `unit_hp` | HP ratio of the detected entity (if any) |
| `unit_face` | Dot product of ray direction with entity's forward vector — encodes whether the target is facing toward/away from the ray |

An **action mask** (6-element binary vector) is also returned alongside the observation. It is used to zero-out logits of invalid actions before the agent samples, ensuring legal moves only.

---

### Action Space

`Discrete(6)` — one of six discrete actions per step:

| ID | Action |
|----|--------|
| 0 | No-op |
| 1 | Move forward |
| 2 | Move backward (half speed; disabled by default via `ALLOW_BACKWARD=False`) |
| 3 | Rotate left |
| 4 | Rotate right |
| 5 | Attack (fires a beam; only valid when an enemy is in range) |

---

### Reward System

Rewards are shaped to encourage aggressive but precise tactical play:

| Signal | Value | Description |
|--------|-------|-------------|
| Step penalty | `+0.03` | Small positive step reward to discourage passive play |
| Damage dealt | `dmg × 2.5` | Scaled reward per HP dealt to enemies |
| Backstab bonus | `× 1.1` | Multiplier on damage reward when attacking from flanks/rear |
| Execute bonus | Up to `× 1.8` | Higher reward for finishing low-HP targets |
| Damage taken | `-dmg × 1.0 × fear` | Penalty for receiving damage; scales up as own HP drops |
| Friendly fire | `-dmg × 5000` | Massive penalty for hitting a teammate |
| Death | `-50` | Penalty when a Pokémon is eliminated |
| Win | `+200` | Reward for surviving team at episode end |
| Loss | `-100` | Penalty for losing team |
| Timeout | `-1000` | Applied to all agents if the episode reaches `MAX_STEPS_PER_EPISODE` |

---

### Neural Network Model

`model.py` defines `PokemonAgent`, an Actor-Critic network:

```
Input (obs_dim=98)
    → Linear(98 → 512) + LayerNorm + LeakyReLU
    → Linear(512 → 256) + LayerNorm + LeakyReLU
    → Linear(256 → 256) + LayerNorm + LeakyReLU
    → Linear(256 → 128) + LayerNorm + LeakyReLU
    ├──→ Actor head: Linear(128 → 6)  [policy logits]
    └──→ Critic head: Linear(128 → 1) [state value]
```

- All layers are initialized with **orthogonal initialization** (std=√2 for hidden layers, std=0.01 for actor, std=1.0 for critic)
- A **single shared model** is used for all agents (parameter sharing across teams), which speeds up training dramatically in multi-agent settings
- During inference, **action masking** sets logits of invalid actions to `-1e9` before sampling from the Categorical distribution

**Training algorithm:** PPO with GAE, clipped surrogate objective, value function clipping, entropy bonus, and gradient norm clipping. All hyperparameters are in `config.py`.

---

## Pokémon Roster

Six species are available, each with a unique 3D model and particle color:

| Species | Color | Particle Color |
|---------|-------|----------------|
| Pikachu | Yellow | Electric yellow |
| Charmander | Red-orange | Fire orange |
| Squirtle | Blue | Water blue |
| Bulbasaur | Green | Grass green |
| Rhyhorn | Grey | Rock grey-brown |
| Eevee | Brown | Normal brown |

> All species currently have identical stats (100 HP, 7 attack power). The differentiation is purely visual. Stats can be individually tuned in `pokemon_data.py`.

---

## Getting Started

### Prerequisites

- Python **3.10+** (tested on 3.12)
- A GPU is optional but recommended for training; inference runs fine on CPU
- The `.glb` asset files must be present in the `assets/` directory

### Installation

```bash
# 1. Clone the repository
git clone git clone -b final --single-branch https://github.com/LegendXAnurag/PokemonShowDown.git
cd PokemonShowDown

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Project

#### Watch AI vs AI (pre-trained model)
```bash
python test.py
```

Optionally override the team configuration:
```bash
python test.py --teams 1v1
python test.py --teams 3v3
python test.py --model-path checkpoints/my_model.pt
```

#### Play as Human vs AI
```bash
python test_human.py
```

A species-selection screen will appear. Use **← →** (or **A / D**) to browse, then **SPACE / ENTER** to confirm your Pokémon.

Optionally configure teams:
```bash
python test_human.py --teams 2v2 --seed 7
```

#### Train a new model
```bash
python train.py
python train.py --render          # show the environment during training
python train.py --resume checkpoints/pokemon_team_battle_2v2_backstab_feb6.pt
```

#### Open the developer testbench (no AI)
```bash
python testbench.py
```

#### Verify spawn distance constraints
```bash
python verify_dist.py
```

---

## Controls

### AI Viewer (`test.py`)

| Key | Action |
|-----|--------|
| `SPACE` / `R` | Reset battle (clears win counter) |
| `C` | Cycle camera: Overhead → Orbit |
| `O` | Toggle Orbit camera on/off |

### Human Mode (`test_human.py`)

| Key / Input | Action |
|-------------|--------|
| `W` | Move forward |
| `A` / `D` | Rotate left / right |
| **Mouse** | Aim (rotates player to face cursor) |
| `SPACE` / **Left Click** | Attack |
| `C` | Cycle camera: Overhead → Follow → Orbit |
| `O` | Toggle Orbit camera |
| `R` | Restart round (after game over) |

### Testbench (`testbench.py`)

| Key | Action |
|-----|--------|
| `W` / `S` | Move forward / backward |
| `A` / `D` | Rotate left / right |
| `SPACE` | Attack |
| `TAB` | Switch active Pokémon |
| `V` | Cycle camera views |
| `1`–`4` | Jump to specific camera view |

---

## Configuration

All simulation parameters are centralized in `config.py`. Key settings:

```python
# Arena
SCREEN_WIDTH  = 1280
SCREEN_HEIGHT = 640
FPS           = 60
GRID_SIZE     = 10       # 10×10 world-unit arena
BOUNDARY      = 5.0      # Arena half-size (-5 to +5)

# Teams
TEAMS_SETUP = [
    {"count": 2, "color": (0.85, 0.1, 0.1)},   # Team 0: Red
    {"count": 2, "color": (0.1, 0.1, 0.85)},    # Team 1: Blue
]

# Combat
ATTACK_RANGE = 2.0
ATTACK_WIDTH = 1.0
MOVE_SPEED   = 0.10
ROTATION_SPEED = 4.0

# Training
LEARNING_RATE       = 3e-4
TOTAL_TIMESTEPS     = 9_000_000
BATCH_SIZE          = 512
GAMMA               = 0.99
GAE_LAMBDA          = 0.95
CLIP_COEF           = 0.2
```

To add a new team or change team sizes, edit `TEAMS_SETUP` in `config.py`, or pass `--teams NvN` at runtime.

---

## Training

Training runs a standard PPO loop using a single shared neural network for all agents. Progress is tracked with a `tqdm` progress bar, and per-update stats (mean reward, value loss, entropy, FPS) are printed to console and saved to `logs/<run_name>.csv`.

Checkpoints are saved to `checkpoints/` every `CHECKPOINT_FREQ` (2500) steps and at the end of training. To resume:

```bash
python train.py --resume checkpoints/pokemon_team_battle_2v2_backstab_feb6.pt
```

**Key training hyperparameters** (all in `config.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TOTAL_TIMESTEPS` | 9,000,000 | Total env steps to train for |
| `NUM_STEPS` | 256 | Rollout steps per update |
| `MINIBATCH_SIZE` | 64 | Mini-batch size for gradient updates |
| `UPDATE_EPOCHS` | 10 | PPO update epochs per rollout |
| `LEARNING_RATE` | 3e-4 | Adam optimizer learning rate |
| `CLIP_COEF` | 0.2 | PPO clipping coefficient (ε) |
| `ENT_COEF` | 0.01 | Entropy bonus coefficient |
| `MAX_GRAD_NORM` | 0.5 | Gradient norm clipping threshold |

---

## File Reference

| File | Purpose |
|------|---------|
| `config.py` | Central configuration — all constants, hyperparameters, reward scales |
| `custom_env.py` | PettingZoo `ParallelEnv` — observations, step logic, reward computation, rendering |
| `model.py` | `PokemonAgent` actor-critic neural network |
| `pokemon.py` | `Pokemon` class — position, movement, attack beam, hitbox, OpenGL drawing |
| `pokemon_data.py` | Static species database (HP, attack, model path, color) |
| `ground.py` | OpenGL ground quad + grid lines + semi-transparent arena walls |
| `hud.py` | `ScoreHUD` — renders win-count text as a 2D pixel overlay via `glWindowPos2i` |
| `particles.py` | `ParticleSystem` — beam trail + hit-burst particles with per-species colors |
| `utils.py` | `ModelLoader` — loads `.glb` files with `trimesh`, centers, scales, draws via OpenGL |
| `train.py` | MAPPO training loop with SuperSuit vectorization, CSV logging, checkpointing |
| `test.py` | Headless AI-vs-AI viewer with orbit camera, HUD, particle FX |
| `test_human.py` | Human-vs-AI mode with species picker, mouse aiming, follow camera |
| `testbench.py` | Developer sandbox: 2 Pokémon, no AI, manual WASD control |
| `verify_dist.py` | Unit test for spawn distance constraint logic |
