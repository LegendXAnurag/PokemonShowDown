# test.py  — AI vs AI Pokemon Battle Viewer
# Added: Particle FX, On-Screen Win Tracking, Orbit Camera, CLI --teams
import argparse, os, math, random, time
import numpy as np
import torch
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

import config
import pokemon_data
from ground import draw_ground, draw_walls
from pokemon import Pokemon
from model import PokemonAgent
from particles import ParticleSystem
from hud import ScoreHUD

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Trained Pokemon Agents (Team Battle)")
    parser.add_argument("--model-path", type=str,
                        default=f"{config.CHECKPOINT_DIR}/{config.MODEL_NAME}",
                        help="path to the trained model checkpoint")
    parser.add_argument("--seed",  type=int, default=1,  help="random seed")
    parser.add_argument("--teams", type=str, default=None,
                        help="Team composition e.g. '2v2', '1v1', '3v3'")
    return parser.parse_args()


def apply_team_config(teams_str):
    PALETTE = [
        (0.85, 0.1,  0.1), (0.1,  0.1,  0.85),
        (0.1,  0.8,  0.1), (0.85, 0.75, 0.0),
    ]
    try:
        counts = [int(x) for x in teams_str.split("v")]
    except ValueError:
        print(f"[WARN] Invalid --teams '{teams_str}'. Using default config.")
        return
    config.TEAMS_SETUP = [{"count": c, "color": PALETTE[i % len(PALETTE)]}
                          for i, c in enumerate(counts)]
    config.NUM_AGENTS = sum(t["count"] for t in config.TEAMS_SETUP)
    print(f"[Teams] {teams_str}: {config.TEAMS_SETUP}")


# ── OpenGL init ───────────────────────────────────────────────────────────────
def init_gl():
    glClearColor(0.53, 0.81, 0.92, 1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (10, 15, 10, 0))
    glLightfv(GL_LIGHT0, GL_AMBIENT,  (0.3, 0.3, 0.3, 1.0))
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, config.SCREEN_WIDTH / config.SCREEN_HEIGHT, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)


# ── HP bar ────────────────────────────────────────────────────────────────────
def draw_hp_bar(pokemon, camera_pos):
    glPushMatrix()
    glTranslatef(pokemon.x, pokemon.y + 2.0, pokemon.z)
    cam_x, _, cam_z = camera_pos
    angle_deg = math.degrees(math.atan2(cam_x - pokemon.x, cam_z - pokemon.z))
    glRotatef(angle_deg, 0, 1, 0)

    w = 1.0; h = 0.15
    hp_ratio = max(0.0, pokemon.hp / pokemon.max_hp)
    glDisable(GL_LIGHTING)

    # red background
    glColor3f(1, 0, 0)
    glBegin(GL_QUADS)
    glVertex3f(-w/2, 0, 0); glVertex3f(w/2, 0, 0)
    glVertex3f(w/2,  h, 0); glVertex3f(-w/2, h, 0)
    glEnd()

    # green foreground
    if hp_ratio > 0:
        glColor4f(0, 1, 0, 0.9)
        cw = w * hp_ratio
        glBegin(GL_QUADS)
        glVertex3f(-w/2, 0, 0.05); glVertex3f(-w/2 + cw, 0, 0.05)
        glVertex3f(-w/2 + cw, h, 0.05); glVertex3f(-w/2, h, 0.05)
        glEnd()

    glEnable(GL_LIGHTING)
    glPopMatrix()


# ── Spawn ─────────────────────────────────────────────────────────────────────
def get_n_spawns(n):
    limit = config.BOUNDARY - config.SPAWN_MARGIN
    positions = []
    for _ in range(n):
        for attempt in range(100):
            x = random.uniform(-limit, limit)
            z = random.uniform(-limit, limit)
            if all(math.hypot(x-ex, z-ez) >= config.MIN_SPAWN_DIST
                   for ex, ez, _ in positions):
                positions.append((x, z, random.uniform(0, 360)))
                break
        else:
            positions.append((random.uniform(-limit, limit),
                              random.uniform(-limit, limit), 0))
    return positions


def reset_battle():
    species_list = list(pokemon_data.POKEMON_DB.keys())
    limit = config.BOUNDARY - config.SPAWN_MARGIN
    occupied = []
    num_teams = len(config.TEAMS_SETUP)
    pokemons = []

    for team_idx, team_setup in enumerate(config.TEAMS_SETUP):
        color = team_setup["color"]
        # Team spawn cluster
        radius = (config.BOUNDARY - config.SPAWN_MARGIN) * 0.75
        angle  = (2 * math.pi / num_teams) * team_idx
        cx = radius * math.cos(angle)
        cz = radius * math.sin(angle)

        for _ in range(team_setup["count"]):
            species = random.choice(species_list)
            placed = False
            for attempt in range(100):
                r     = random.uniform(0, config.TEAM_MEMBER_DIST)
                theta = random.uniform(0, 2 * math.pi)
                x = cx + r * math.cos(theta)
                z = cz + r * math.sin(theta)
                if not (-limit <= x <= limit and -limit <= z <= limit):
                    continue
                valid = True
                for ex, ez, et in occupied:
                    dist = math.hypot(x - ex, z - ez)
                    threshold = config.MIN_TEAMMATE_DIST if et == team_idx else config.MIN_SPAWN_DIST
                    if dist < threshold:
                        valid = False; break
                if valid:
                    occupied.append((x, z, team_idx))
                    placed = True
                    break
            if not placed:
                x = random.uniform(-limit, limit)
                z = random.uniform(-limit, limit)
                occupied.append((x, z, team_idx))

            p = Pokemon(species, pokemon_data.POKEMON_DB[species], (x, 0, z), team_idx, color)
            p.angle = random.uniform(0, 360)
            pokemons.append(p)

    return pokemons


# ── AI observation / mask ─────────────────────────────────────────────────────
def build_observation(me, all_pokemons):
    hp_norm = me.hp / me.max_hp
    cd_norm = me.attack_timer / config.ATTACK_DURATION if me.attack_timer > 0 else 0.0
    self_state = [hp_norm, cd_norm]
    lidar_data = []
    angle_step = 360.0 / config.NUM_RAYS
    max_dist   = config.VISION_RANGE
    boundary   = config.BOUNDARY

    for i in range(config.NUM_RAYS):
        ray_angle = (me.angle + i * angle_step) % 360
        rad = math.radians(ray_angle)
        dir_x = math.sin(rad); dir_z = math.cos(rad)

        dist_wall = max_dist
        if abs(dir_x) > 1e-6:
            for t in ((boundary - me.x)/dir_x, (-boundary - me.x)/dir_x):
                if t > 0: dist_wall = min(dist_wall, t)
        if abs(dir_z) > 1e-6:
            for t in ((boundary - me.z)/dir_z, (-boundary - me.z)/dir_z):
                if t > 0: dist_wall = min(dist_wall, t)

        dist_entity = max_dist; hit_entity = False; entity_obj = None
        for target in all_pokemons:
            if target is me or target.hp <= 0: continue
            fc_x = target.x - me.x; fc_z = target.z - me.z
            t_proj = fc_x * dir_x + fc_z * dir_z
            if t_proj <= 0: continue
            cx = me.x + dir_x * t_proj; cz = me.z + dir_z * t_proj
            dsq = (cx - target.x)**2 + (cz - target.z)**2
            rsq = config.HITBOX_RADIUS**2
            if dsq < rsq:
                t_hit = t_proj - math.sqrt(rsq - dsq)
                if t_hit < dist_entity:
                    dist_entity = t_hit; hit_entity = True; entity_obj = target

        final_dist = min(dist_wall, dist_entity, max_dist)
        nd = final_dist / max_dist
        ch_wall  = 1.0 if (final_dist == dist_wall and final_dist < max_dist) else 0.0
        ch_enemy = ch_mate = ch_hp = ch_face = 0.0
        if hit_entity and entity_obj is not None and final_dist < dist_wall:
            if entity_obj.team_id == me.team_id: ch_mate = 1.0
            else: ch_enemy = 1.0
            ch_hp = entity_obj.hp / entity_obj.max_hp
            efx, efz = entity_obj.get_forward_vector()
            ch_face = dir_x * efx + dir_z * efz
        lidar_data.extend([nd, ch_wall, ch_enemy, ch_mate, ch_hp, ch_face])

    return np.array(self_state + lidar_data, dtype=np.float32)


def check_collision(me, x, z, all_mons):
    for t in all_mons:
        if t is me or t.hp <= 0: continue
        if (x - t.x)**2 + (z - t.z)**2 < 1.0: return True
    return False


def get_action_mask(pokemon, all_pokemons):
    mask = np.zeros(6, dtype=np.int8)
    if pokemon.hp <= 0 or pokemon.is_attacking:
        mask[0] = 1; return mask
    mask[1] = 1; mask[3] = 1; mask[4] = 1
    mask[2] = 1 if config.ALLOW_BACKWARD else 0
    mask[5] = 1 if pokemon.check_hit(all_pokemons) else 0
    px, pz = pokemon.predict_position(1)
    if check_collision(pokemon, px, pz, all_pokemons): mask[1] = 0
    if config.ALLOW_BACKWARD:
        px, pz = pokemon.predict_position(-1)
        if check_collision(pokemon, px, pz, all_pokemons): mask[2] = 0
    return mask


# ── Camera ────────────────────────────────────────────────────────────────────
def compute_camera(orbit_angle, use_orbit):
    if use_orbit:
        r = 18
        cam_x = r * math.cos(math.radians(orbit_angle))
        cam_z = r * math.sin(math.radians(orbit_angle))
        return (cam_x, 14, cam_z), (0, 0, 0)
    return (0, 25, 20), (0, 0, 0)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    if args.teams:
        apply_team_config(args.teams)

    pygame.init()
    pygame.font.init()
    pygame.display.set_mode(
        (config.SCREEN_WIDTH, config.SCREEN_HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption(
        f"AI vs AI ({config.NUM_AGENTS} agents) | SPACE: Reset | O: Orbit Camera")
    init_gl()

    # Fix BUG-07: restore correct CUDA detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    obs_dim    = 2 + config.NUM_RAYS * config.LIDAR_CHANNELS
    action_dim = 6
    agent = PokemonAgent(obs_dim=obs_dim, action_dim=action_dim).to(device)

    if os.path.exists(args.model_path):
        ckpt = torch.load(args.model_path, map_location=device)
        agent.load_state_dict(ckpt["model_state_dict"])
        print(f"[Model] Loaded from {args.model_path}")
    else:
        print(f"[Model] Not found at {args.model_path}. Using random weights.")
    agent.eval()

    random.seed(args.seed)
    pokemons = reset_battle()

    # Win tracking
    win_counts = {i: 0 for i in range(len(config.TEAMS_SETUP))}
    score_hud  = ScoreHUD()
    score_hud.update(win_counts)

    # Particle system
    particles  = ParticleSystem()

    # Camera state
    orbit_angle = 0.0
    use_orbit   = False

    clock   = pygame.time.Clock()
    running = True

    while running:
        dt = clock.tick(config.FPS) / 1000.0

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pokemons  = reset_battle()
                    particles = ParticleSystem()
                if event.key == pygame.K_o:
                    use_orbit = not use_orbit

        # ── Timers ────────────────────────────────────────────────────────────
        for p in pokemons:
            p.update_timers(config.DT)

        # ── Orbit angle ───────────────────────────────────────────────────────
        if use_orbit:
            orbit_angle = (orbit_angle + 18 * dt) % 360

        # ── Camera ────────────────────────────────────────────────────────────
        cam_pos, look_at = compute_camera(orbit_angle, use_orbit)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluLookAt(*cam_pos, *look_at, 0, 1, 0)

        # ── AI actions ────────────────────────────────────────────────────────
        obs_list  = []
        mask_list = []
        for p in pokemons:
            obs_list.append(build_observation(p, pokemons))
            mask_list.append(get_action_mask(p, pokemons))

        obs_t  = torch.tensor(np.array(obs_list),  dtype=torch.float32).to(device)
        mask_t = torch.tensor(np.array(mask_list), dtype=torch.int8).to(device)
        with torch.no_grad():
            actions, _, _, _ = agent.get_action_and_value(obs_t, action_mask=mask_t)
            actions = actions.cpu().numpy()

        for i, p in enumerate(pokemons):
            if p.hp <= 0: continue
            act = actions[i]
            if not p.is_attacking:
                if   act == 1: p.move_forward()
                elif act == 2: p.move_forward(speed=-config.MOVE_SPEED / 2)
                elif act == 3: p.rotate(-1)
                elif act == 4: p.rotate(1)
                elif act == 5:
                    hit = p.attack(pokemons)
                    if hit:  # particles only on successful enemy hit
                        particles.emit_attack(p, p.actual_beam_length, True)

        # ── Win check ─────────────────────────────────────────────────────────
        alive_teams = {p.team_id for p in pokemons if p.hp > 0}
        n_teams = len(config.TEAMS_SETUP)
        game_over = (len(alive_teams) <= 1) if n_teams > 1 else (len(alive_teams) == 0)

        if game_over:
            if len(alive_teams) == 1:
                w = list(alive_teams)[0]
                win_counts[w] = win_counts.get(w, 0) + 1
                label = f"TEAM {w} WON!"
            else:
                label = "DRAW!"
            score_str = "  |  ".join(f"T{t}: {win_counts[t]}W" for t in win_counts)
            pygame.display.set_caption(f"{label}  [{score_str}]  | O: Orbit | SPACE: Reset")
            score_hud.update(win_counts)
            print(f"[Result] {label}  |  {score_str}")

            # Draw final frame, pause briefly, then restart
            draw_ground(); draw_walls()
            for p in pokemons:
                if p.hp > 0:
                    p.draw(); draw_hp_bar(p, cam_pos)
            particles.draw()
            score_hud.draw(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)
            pygame.display.flip()
            time.sleep(1.5)
            pokemons  = reset_battle()
            particles = ParticleSystem()
            continue

        # ── Draw ──────────────────────────────────────────────────────────────
        draw_ground()
        draw_walls()
        particles.update(dt)

        for p in pokemons:
            if p.hp > 0:
                p.draw()
                draw_hp_bar(p, cam_pos)

        particles.draw()
        score_hud.draw(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()