# test_human.py  — Human vs AI Pokemon Battle
# Features: Species Selection, Win Tracking, Particle FX, Follow/Orbit Camera, CLI Teams
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

# ── Camera modes ──────────────────────────────────────────────────────────────
CAMERA_OVERHEAD = "overhead"
CAMERA_FOLLOW   = "follow"
CAMERA_ORBIT    = "orbit"

# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Human vs AI Pokemon Battle")
    p.add_argument("--model-path", type=str,
                   default=f"{config.CHECKPOINT_DIR}/{config.MODEL_NAME}")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--teams", type=str, default=None,
                   help="Team composition e.g. '2v2', '1v1', '3v3'")
    return p.parse_args()


def apply_team_config(teams_str):
    """Parse '2v2' / '1v3' etc. and override config at runtime."""
    PALETTE = [
        (0.85, 0.1,  0.1),   # red
        (0.1,  0.1,  0.85),  # blue
        (0.1,  0.8,  0.1),   # green
        (0.85, 0.75, 0.0),   # yellow
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


# ── Species selection (2D pygame window, before OpenGL init) ──────────────────
def show_species_selection():
    """Show a pure-2D pygame selection screen. Returns chosen species key."""
    W, H = 920, 460
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Choose Your Pokemon!  ← → to browse  |  SPACE to confirm")

    species_list = list(pokemon_data.POKEMON_DB.keys())
    idx = 0
    clock = pygame.time.Clock()

    BG       = (12,  12,  30)
    TITLE_C  = (255, 230, 60)
    SUB_C    = (170, 170, 200)
    WHITE    = (255, 255, 255)
    BORDER_C = (255, 255, 100)

    font_title = pygame.font.Font(None, 62)
    font_sub   = pygame.font.Font(None, 30)
    font_name  = pygame.font.Font(None, 34)
    font_big   = pygame.font.Font(None, 46)

    CARD_W, CARD_H = 118, 72
    GAP = 18

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_LEFT,  pygame.K_a):
                    idx = (idx - 1) % len(species_list)
                if event.key in (pygame.K_RIGHT, pygame.K_d):
                    idx = (idx + 1) % len(species_list)
                if event.key in (pygame.K_RETURN, pygame.K_SPACE):
                    pygame.display.quit()
                    return species_list[idx]

        screen.fill(BG)

        # Title
        t = font_title.render("Choose Your Pokemon!", True, TITLE_C)
        screen.blit(t, (W//2 - t.get_width()//2, 28))
        s = font_sub.render("← / → to browse    SPACE or ENTER to confirm", True, SUB_C)
        screen.blit(s, (W//2 - s.get_width()//2, 88))

        # Cards
        total_w = len(species_list) * (CARD_W + GAP) - GAP
        start_x = W//2 - total_w//2

        for i, sp in enumerate(species_list):
            data = pokemon_data.POKEMON_DB[sp]
            cf   = data["color_fallback"]
            card_col  = (int(cf[0]*200), int(cf[1]*200), int(cf[2]*200))
            dark_col  = (int(cf[0]*90),  int(cf[1]*90),  int(cf[2]*90))
            x = start_x + i * (CARD_W + GAP)
            y = H//2 - CARD_H//2 - 10

            if i == idx:
                pygame.draw.rect(screen, BORDER_C, (x-4, y-4, CARD_W+8, CARD_H+8), border_radius=10)
                pygame.draw.rect(screen, card_col,  (x,   y,   CARD_W,   CARD_H),   border_radius=8)
            else:
                pygame.draw.rect(screen, dark_col,  (x,   y,   CARD_W,   CARD_H),   border_radius=8)

            n = font_name.render(data["name"], True, WHITE)
            screen.blit(n, (x + CARD_W//2 - n.get_width()//2, y + CARD_H//2 - n.get_height()//2))

        # Selected info
        sel = font_big.render(f"▶  {pokemon_data.POKEMON_DB[species_list[idx]]['name']}  ◀", True, TITLE_C)
        screen.blit(sel, (W//2 - sel.get_width()//2, H - 100))

        pygame.display.flip()
        clock.tick(60)


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


# ── Spawn helpers ─────────────────────────────────────────────────────────────
def get_n_spawns(n):
    limit = config.BOUNDARY - config.SPAWN_MARGIN
    spawn_radius = config.HITBOX_RADIUS * 2.5
    positions = []
    for i in range(n):
        spawned = False
        for _ in range(500):
            x = random.uniform(-limit, limit)
            z = random.uniform(-limit, limit)
            if all(math.hypot(x-ex, z-ez) >= spawn_radius for ex, ez, _ in positions):
                positions.append((x, z, random.uniform(0, 360)))
                spawned = True
                break
        if not spawned:
            print(f"[WARN] Forced spawn for agent {i}")
            positions.append((random.uniform(-limit, limit), random.uniform(-limit, limit), 0))
    return positions


def reset_battle(player_species=None):
    n = config.NUM_AGENTS
    spawns = get_n_spawns(n)
    pokemons = []
    species_list = list(pokemon_data.POKEMON_DB.keys())
    global_idx = 0
    player_assigned = False

    for team_idx, team_setup in enumerate(config.TEAMS_SETUP):
        color = team_setup["color"]
        for _ in range(team_setup["count"]):
            if global_idx >= len(spawns):
                break
            x, z, rot = spawns[global_idx]

            if team_idx == 0 and not player_assigned and player_species:
                species = player_species
                player_assigned = True
            else:
                species = random.choice(species_list)

            p = Pokemon(species, pokemon_data.POKEMON_DB[species], (x, 0, z), team_idx, color)
            p.angle = rot
            if team_idx == 0 and not any(getattr(q, "is_human", False) for q in pokemons):
                p.is_human = True
            pokemons.append(p)
            global_idx += 1

    return pokemons


# ── Mouse raycasting ──────────────────────────────────────────────────────────
def get_mouse_ground_intersection(mx, my):
    mv  = glGetDoublev(GL_MODELVIEW_MATRIX)
    prj = glGetDoublev(GL_PROJECTION_MATRIX)
    vp  = glGetIntegerv(GL_VIEWPORT)
    gl_y = vp[3] - my
    try:
        near = gluUnProject(mx, gl_y, 0.0, mv, prj, vp)
        far  = gluUnProject(mx, gl_y, 1.0, mv, prj, vp)
    except Exception:
        return None
    ray_o = np.array(near)
    ray_d = np.array(far) - ray_o
    if abs(ray_d[1]) < 1e-6:
        return None
    t = -ray_o[1] / ray_d[1]
    if t < 0:
        return None
    pt = ray_o + t * ray_d
    return float(pt[0]), float(pt[2])


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
            t_proj = fc_x*dir_x + fc_z*dir_z
            if t_proj <= 0: continue
            cx = me.x + dir_x*t_proj; cz = me.z + dir_z*t_proj
            dsq = (cx-target.x)**2 + (cz-target.z)**2
            rsq = config.HITBOX_RADIUS**2
            if dsq < rsq:
                t_hit = t_proj - math.sqrt(rsq - dsq)
                if t_hit < dist_entity:
                    dist_entity = t_hit; hit_entity = True; entity_obj = target

        final_dist = min(dist_wall, dist_entity, max_dist)
        nd = final_dist / max_dist
        ch_wall = 1.0 if (final_dist == dist_wall and final_dist < max_dist) else 0.0
        ch_enemy = ch_mate = ch_hp = ch_face = 0.0
        if hit_entity and entity_obj is not None and final_dist < dist_wall:
            if entity_obj.team_id == me.team_id: ch_mate = 1.0
            else: ch_enemy = 1.0
            ch_hp   = entity_obj.hp / entity_obj.max_hp
            efx, efz = entity_obj.get_forward_vector()
            ch_face = dir_x*efx + dir_z*efz
        lidar_data.extend([nd, ch_wall, ch_enemy, ch_mate, ch_hp, ch_face])

    return np.array(self_state + lidar_data, dtype=np.float32)


def check_collision(me, x, z, all_mons):
    min_dsq = 1.0
    for t in all_mons:
        if t is me or t.hp <= 0: continue
        if (x-t.x)**2 + (z-t.z)**2 < min_dsq: return True
    return False


def get_action_mask(pokemon, all_pokemons):
    mask = np.zeros(6, dtype=np.int8)
    if pokemon.hp <= 0 or pokemon.is_attacking:
        mask[0] = 1
        return mask
    mask[1] = 1; mask[3] = 1; mask[4] = 1
    mask[2] = 1 if config.ALLOW_BACKWARD else 0
    mask[5] = 1 if pokemon.check_hit(all_pokemons) else 0
    px, pz = pokemon.predict_position(1)
    if check_collision(pokemon, px, pz, all_pokemons): mask[1] = 0
    if config.ALLOW_BACKWARD:
        px, pz = pokemon.predict_position(-1)
        if check_collision(pokemon, px, pz, all_pokemons): mask[2] = 0
    return mask


# ── HP bar / indicators ───────────────────────────────────────────────────────
def draw_indicators(pokemon, camera_pos, is_player=False):
    glPushMatrix()
    glTranslatef(pokemon.x, pokemon.y + 2.5, pokemon.z)
    cam_x, _, cam_z = camera_pos
    angle_deg = math.degrees(math.atan2(cam_x - pokemon.x, cam_z - pokemon.z))
    glRotatef(angle_deg, 0, 1, 0)

    if is_player:
        glPushMatrix()
        glTranslatef(0, 0.6, 0); glScalef(0.4, 0.4, 0.4)
        glDisable(GL_LIGHTING)
        glColor3f(0.2, 1.0, 0.2)
        glBegin(GL_TRIANGLES)
        glVertex3f(-0.5, 0.5, 0); glVertex3f(0.5, 0.5, 0); glVertex3f(0, -0.5, 0)
        glEnd()
        glEnable(GL_LIGHTING)
        glPopMatrix()

    w = 1.0; h = 0.15
    hp_ratio = max(0.0, pokemon.hp / pokemon.max_hp)
    glDisable(GL_LIGHTING)
    glColor3f(0, 0, 0); glLineWidth(2.0)
    glBegin(GL_LINE_LOOP)
    glVertex3f(-w/2,0,0); glVertex3f(w/2,0,0); glVertex3f(w/2,h,0); glVertex3f(-w/2,h,0)
    glEnd(); glLineWidth(1.0)
    if hp_ratio > 0:
        cw = w * hp_ratio
        glColor3f(0.0, 0.35, 0.0)
        glBegin(GL_QUADS)
        glVertex3f(-w/2,0,0); glVertex3f(-w/2+cw,0,0)
        glVertex3f(-w/2+cw,h,0); glVertex3f(-w/2,h,0)
        glEnd()
    glEnable(GL_LIGHTING)
    glPopMatrix()


# ── 2D HUD: win counts in the window caption (reliable cross-platform) ────────
def update_caption_score(player_species, win_counts):
    pname = pokemon_data.POKEMON_DB[player_species]["name"]
    score_str = "  |  ".join(f"Team {t}: {w}W" for t, w in sorted(win_counts.items()))
    pygame.display.set_caption(
        f"YOU ({pname}) vs AI   [{score_str}]   "
        f"WASD+Mouse+Space  |  C:Camera  |  O:Orbit  |  R:Reset")


# ── Camera helpers ────────────────────────────────────────────────────────────
def compute_camera(mode, player_p, orbit_angle):
    """Returns (cam_x, cam_y, cam_z, look_x, look_y, look_z)."""
    if mode == CAMERA_FOLLOW and player_p and player_p.hp > 0:
        rad = math.radians(player_p.angle)
        fx = math.sin(rad); fz = math.cos(rad)
        cam_x = player_p.x - fx * 5
        cam_y = player_p.y + 4
        cam_z = player_p.z - fz * 5
        return cam_x, cam_y, cam_z, player_p.x, player_p.y + 0.5, player_p.z
    elif mode == CAMERA_ORBIT:
        r = 18
        cam_x = r * math.cos(math.radians(orbit_angle))
        cam_z = r * math.sin(math.radians(orbit_angle))
        return cam_x, 14, cam_z, 0, 0, 0
    else:  # CAMERA_OVERHEAD (default)
        if player_p and player_p.hp > 0:
            cam_x = player_p.x
            cam_z = player_p.z + 8
            return cam_x, 10, cam_z, player_p.x, 0, player_p.z
        return 0, 25, 20, 0, 0, 0


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    if args.teams:
        apply_team_config(args.teams)

    # FEATURE-04: Species selection (2D screen before OpenGL)
    player_species = show_species_selection()
    print(f"[Player] Chose {pokemon_data.POKEMON_DB[player_species]['name']}")

    # Now open OpenGL window
    pygame.init()
    pygame.font.init()
    pygame.display.set_mode(
        (config.SCREEN_WIDTH, config.SCREEN_HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption(
        f"YOU ({pokemon_data.POKEMON_DB[player_species]['name']}) vs AI  |  "
        f"WASD+Mouse+Space  |  C: Camera  |  O: Orbit")
    init_gl()

    # Load model
    device = torch.device("cpu")
    obs_dim = 2 + config.NUM_RAYS * config.LIDAR_CHANNELS
    agent = PokemonAgent(obs_dim=obs_dim, action_dim=6).to(device)
    if os.path.exists(args.model_path):
        ckpt = torch.load(args.model_path, map_location=device)
        agent.load_state_dict(ckpt["model_state_dict"])
        print(f"[Model] Loaded from {args.model_path}")
    else:
        print(f"[Model] Not found at {args.model_path} — AI uses random weights.")
    agent.eval()

    random.seed(args.seed)
    pokemons = reset_battle(player_species)

    # FEATURE-06: win tracking
    win_counts = {i: 0 for i in range(len(config.TEAMS_SETUP))}

    # FEATURE-07: particle system
    particles = ParticleSystem()

    # Score HUD (visible on-screen)
    score_hud = ScoreHUD()
    score_hud.update(win_counts)

    # Camera state (FEATURE-08 / FEATURE-10)
    camera_mode  = CAMERA_OVERHEAD
    orbit_angle  = 0.0          # degrees, slowly increments in orbit mode

    clock = pygame.time.Clock()
    running   = True
    game_over = False

    # Show initial score in caption
    update_caption_score(player_species, win_counts)

    while running:
        dt = clock.tick(config.FPS) / 1000.0  # seconds

        # ── Events ────────────────────────────────────────────────────────────
        attack_command = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and game_over:
                    pokemons   = reset_battle(player_species)
                    particles  = ParticleSystem()
                    game_over  = False
                    update_caption_score(player_species, win_counts)
                if event.key == pygame.K_SPACE:
                    attack_command = True
                # FEATURE-08: cycle camera
                if event.key == pygame.K_c:
                    modes = [CAMERA_OVERHEAD, CAMERA_FOLLOW, CAMERA_ORBIT]
                    camera_mode = modes[(modes.index(camera_mode) + 1) % len(modes)]
                # FEATURE-10: quick orbit toggle
                if event.key == pygame.K_o:
                    camera_mode = CAMERA_ORBIT if camera_mode != CAMERA_ORBIT else CAMERA_OVERHEAD
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                attack_command = True

        # ── Find player ───────────────────────────────────────────────────────
        player_p = next((p for p in pokemons if getattr(p, "is_human", False)), None)

        # ── Update timers ─────────────────────────────────────────────────────
        for p in pokemons:
            p.update_timers(config.DT)

        # ── Orbit angle ───────────────────────────────────────────────────────
        if camera_mode == CAMERA_ORBIT:
            orbit_angle = (orbit_angle + 18 * dt) % 360   # full orbit ~20 s

        # ── Camera & clear ────────────────────────────────────────────────────
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        cam_x, cam_y, cam_z, lx, ly, lz = compute_camera(camera_mode, player_p, orbit_angle)
        gluLookAt(cam_x, cam_y, cam_z, lx, ly, lz, 0, 1, 0)

        # ── Game logic ────────────────────────────────────────────────────────
        if not game_over:
            # Human
            if player_p and player_p.hp > 0 and not player_p.is_attacking:
                if attack_command:
                    hit = player_p.attack(pokemons)
                    if hit:  # particles only on successful enemy hit
                        particles.emit_attack(player_p, player_p.actual_beam_length, True)
                else:
                    # Mouse rotation
                    mx, my = pygame.mouse.get_pos()
                    gp = get_mouse_ground_intersection(mx, my)
                    if gp:
                        dx = gp[0] - player_p.x; dz = gp[1] - player_p.z
                        if dx*dx + dz*dz > 0.01:
                            player_p.angle = math.degrees(math.atan2(dx, dz))
                    keys = pygame.key.get_pressed()
                    if keys[pygame.K_a]: player_p.rotate(-1)
                    if keys[pygame.K_d]: player_p.rotate(1)
                    if keys[pygame.K_w]:
                        px, pz = player_p.predict_position(1)
                        if not check_collision(player_p, px, pz, pokemons):
                            player_p.move_forward()

            # AI
            ai_idx_list, obs_list, mask_list = [], [], []
            for i, p in enumerate(pokemons):
                if getattr(p, "is_human", False) or p.hp <= 0: continue
                obs_list.append(build_observation(p, pokemons))
                mask_list.append(get_action_mask(p, pokemons))
                ai_idx_list.append(i)

            if obs_list:
                obs_t  = torch.tensor(np.array(obs_list),  dtype=torch.float32).to(device)
                mask_t = torch.tensor(np.array(mask_list), dtype=torch.int8).to(device)
                with torch.no_grad():
                    actions, _, _, _ = agent.get_action_and_value(obs_t, action_mask=mask_t)
                    actions = actions.cpu().numpy()
                for idx, act in zip(ai_idx_list, actions):
                    p = pokemons[idx]
                    if not p.is_attacking:
                        if   act == 1: p.move_forward()
                        elif act == 2: p.move_forward(speed=-config.MOVE_SPEED/2)
                        elif act == 3: p.rotate(-1)
                        elif act == 4: p.rotate(1)
                        elif act == 5:
                            hit = p.attack(pokemons)
                            if hit:  # particles only on successful enemy hit
                                particles.emit_attack(p, p.actual_beam_length, True)

            # Win check
            alive_teams = {p.team_id for p in pokemons if p.hp > 0}
            n_teams = len(config.TEAMS_SETUP)
            ended = (len(alive_teams) <= 1) if n_teams > 1 else (len(alive_teams) == 0)
            if ended:
                game_over = True
                if len(alive_teams) == 1:
                    w = list(alive_teams)[0]
                    win_counts[w] = win_counts.get(w, 0) + 1
                    label = "YOU WON! (R)" if w == 0 else f"TEAM {w} WON! (R)"
                else:
                    label = "DRAW! (R)"
                # FEATURE-06: caption + visible HUD
                score_str = "  |  ".join(f"T{t}: {win_counts[t]}W" for t in win_counts)
                pygame.display.set_caption(f"{label}   [{score_str}]")
                score_hud.update(win_counts)
                print(f"[Result] {label}  |  {score_str}")

        # ── Particles update ──────────────────────────────────────────────────
        particles.update(dt)

        # ── Draw scene ────────────────────────────────────────────────────────
        draw_ground()
        draw_walls()

        for p in pokemons:
            if p.hp > 0:
                p.draw()
                draw_indicators(p, (cam_x, cam_y, cam_z), is_player=getattr(p, "is_human", False))

        # FEATURE-07: draw particles on top
        particles.draw()

        # Score HUD overlay (always visible)
        score_hud.draw(config.SCREEN_WIDTH, config.SCREEN_HEIGHT)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()