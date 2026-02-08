import random
import math
import numpy as np

# Mock Config
BOUNDARY = 5.0
SPAWN_MARGIN = 1.0
MIN_SPAWN_DIST = 4.0
TEAM_MEMBER_DIST = 1.5
MIN_TEAMMATE_DIST = 1.5 # The new constraint

TEAMS_SETUP = [
    {"count": 2},
    {"count": 2},
]
NUM_AGENTS = 4

def get_team_center(team_idx, total_teams):
    radius = (BOUNDARY - SPAWN_MARGIN) * 0.75 
    angle = (2 * math.pi / total_teams) * team_idx
    cx = radius * math.cos(angle)
    cz = radius * math.sin(angle)
    return cx, cz

def test_spawn():
    limit = BOUNDARY - SPAWN_MARGIN
    positions = []
    team_map = {}
    
    agent_idx = 0
    for team_id, team_setup in enumerate(TEAMS_SETUP):
        cx, cz = get_team_center(team_id, len(TEAMS_SETUP))
        
        for _ in range(team_setup["count"]):
            attempts = 0
            placed = False
            while attempts < 100:
                # Spawn around team center
                r = random.uniform(0, TEAM_MEMBER_DIST)
                theta = random.uniform(0, 2*math.pi)
                x = cx + r * math.cos(theta)
                z = cz + r * math.sin(theta)
                
                if not (-limit <= x <= limit and -limit <= z <= limit):
                    attempts += 1
                    continue

                valid = True
                for existing_idx, (ex, ez) in enumerate(positions):
                    dist = math.sqrt((x - ex)**2 + (z - ez)**2)
                    existing_team = team_map[existing_idx]
                    
                    if existing_team == team_id:
                        if dist < MIN_TEAMMATE_DIST: # New Check
                            valid = False
                            break
                    else:
                        if dist < MIN_SPAWN_DIST:
                            valid = False
                            break
                
                if valid:
                    positions.append((x, z))
                    team_map[agent_idx] = team_id
                    placed = True
                    break
                attempts += 1
            
            if not placed:
                print(f"Failed to place agent {agent_idx}")
                positions.append((random.uniform(-limit, limit), random.uniform(-limit, limit)))
                team_map[agent_idx] = team_id
            
            agent_idx += 1
            
    # Analyze
    min_team_dist = 100.0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
             dist = math.sqrt((positions[i][0]-positions[j][0])**2 + (positions[i][1]-positions[j][1])**2)
             t1 = team_map[i]
             t2 = team_map[j]
             if t1 == t2:
                 min_team_dist = min(min_team_dist, dist)
                 print(f"Teammate Dist: {dist:.2f}")

    print(f"Min Teammate Dist: {min_team_dist:.2f}")
    if min_team_dist >= MIN_TEAMMATE_DIST - 1e-5:
        print("PASS: Teammate distance constraint respected.")
    else:
        print("FAIL: Teammate distance constraint violated.")

if __name__ == "__main__":
    test_spawn()
