import argparse
import os
import time
import random
import csv  
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import supersuit as ss

import config
from custom_env import PokemonBattleEnv
from model import PokemonAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Train Pokemon Agent using MAPPO")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--cuda", type=bool, default=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--render", action="store_true",
        help="if toggled, render the environment during training")
    parser.add_argument("--resume", type=str, default=None,
        help="path to checkpoint to resume training from")
    args = parser.parse_args()
    return args

def make_env(render_mode=None):
    env = PokemonBattleEnv(render_mode=render_mode)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=0, base_class='gymnasium')
    return env

def main():
    args = parse_args()
    timestamp = int(time.time())
    run_name = f"{args.exp_name}_seed{args.seed}_{timestamp}"
    
    log_dir = "logs"
    if not os.path.exists(log_dir): os.makedirs(log_dir)
    log_file_path = os.path.join(log_dir, f"{run_name}.csv")
    
    # [CHANGE] Generic CSV Headers
    with open(log_file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["update", "global_step", "fps", 
                         "mean_reward", "val_loss", 
                         "total_loss", "entropy", "lr"])
    
    print(f"Logging to: {log_file_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    envs = make_env("human" if args.render else None)

    obs_shape = envs.observation_space['observation'].shape[0]
    action_shape = envs.action_space.n
    print(f"Observation Shape: {obs_shape} | Action Shape: {action_shape}")
    
    agent = PokemonAgent(obs_dim=obs_shape, action_dim=action_shape).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=config.LEARNING_RATE, eps=1e-5)

    global_step = 0
    start_update = 1
    start_time = time.time()
    
    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint {args.resume}...")
        ckpt = torch.load(args.resume)
        agent.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        global_step = ckpt['global_step']
        start_update = (global_step // (config.NUM_STEPS * envs.num_envs)) + 1

    num_envs = envs.num_envs 
    
    obs = torch.zeros((config.NUM_STEPS, num_envs, obs_shape)).to(device)
    masks = torch.zeros((config.NUM_STEPS, num_envs, action_shape)).to(device)
    actions = torch.zeros((config.NUM_STEPS, num_envs) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((config.NUM_STEPS, num_envs)).to(device)
    rewards = torch.zeros((config.NUM_STEPS, num_envs)).to(device)
    dones = torch.zeros((config.NUM_STEPS, num_envs)).to(device)
    values = torch.zeros((config.NUM_STEPS, num_envs)).to(device)

    next_obs_dict, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs_dict['observation']).to(device)
    next_mask = torch.Tensor(next_obs_dict['action_mask']).to(device)
    next_done = torch.zeros(num_envs).to(device)

    # [CHANGE] Aggregate Episode Stats
    episode_rewards = np.zeros(num_envs)
    match_count = 0
    
    batch_size = int(num_envs * config.NUM_STEPS)
    num_updates = config.TOTAL_TIMESTEPS // batch_size

    for update in range(start_update, num_updates + 1):
        
        for step in range(config.NUM_STEPS):
            global_step += num_envs
            obs[step] = next_obs
            masks[step] = next_mask
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, action_mask=next_mask)
                values[step] = value.flatten()
            
            actions[step] = action
            logprobs[step] = logprob

            next_obs_dict, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            
            # [CHANGE] Track N-Agent Match Stats
            episode_rewards += reward
            
            # Check if match ended (if any agent is done, they are all reset in the env implementation)
            if next_done_np[0]: 
                match_count += 1
                
                # Retrieve HP from infos
                hps = []
                for i in range(num_envs):
                    # Handle VectorEnv wrapping
                    info = infos[i]
                    hp = info.get("final_info", info).get("hp", 0)
                    hps.append(hp)
                
                # Determine Survivors
                survivors = [i for i, hp in enumerate(hps) if hp > 0]
                if len(survivors) == 1:
                    winner_text = f"Agent_{survivors[0]}"
                elif len(survivors) == 0:
                    winner_text = "Draw (Everyone Died)"
                else:
                    winner_text = "Time Limit (Draw)"

                print(f" >> Match {match_count} Finished | Winner: {winner_text}")
                print(f"    Avg Ep Reward: {np.mean(episode_rewards):.2f}")
                
                episode_rewards = np.zeros(num_envs)

            next_obs = torch.Tensor(next_obs_dict['observation']).to(device)
            next_mask = torch.Tensor(next_obs_dict['action_mask']).to(device)
            next_done = torch.Tensor(next_done_np).to(device)
            
            if args.render: envs.render()

        # --- GAE ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config.NUM_STEPS)):
                if t == config.NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config.GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config.GAMMA * config.GAE_LAMBDA * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1, obs_shape))
        b_masks = masks.reshape((-1, action_shape))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(batch_size)
        
        # [CHANGE] Simplified logging logic (Mean across batch)
        with torch.no_grad():
            _, _, _, v_pred = agent.get_action_and_value(b_obs, b_actions.long(), action_mask=b_masks)
            val_loss_batch = 0.5 * ((v_pred.view(-1) - b_returns) ** 2).mean().item()
            mean_rew_batch = rewards.mean().item()

        # --- UPDATE ---
        for epoch in range(config.UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, config.MINIBATCH_SIZE):
                end = start + config.MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds], action_mask=b_masks[mb_inds]
                )
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.CLIP_COEF, 1 + config.CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -config.CLIP_COEF, config.CLIP_COEF)
                v_loss = 0.5 * torch.max(v_loss_unclipped, ((v_clipped - b_returns[mb_inds]) ** 2)).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ENT_COEF * entropy_loss + v_loss * config.VF_COEF

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()

        # --- LOGGING ---
        fps = int(global_step / (time.time() - start_time))
        print(f"Update {update}/{num_updates} | Step {global_step} | FPS: {fps}")
        print(f"  > Mean Reward: {mean_rew_batch:.2f}")
        print(f"  > V-Loss: {val_loss_batch:.3f} | Total Loss={loss.item():.3f}")

        with open(log_file_path, mode='a', newline='') as f:
            csv.writer(f).writerow([update, global_step, fps, 
                                    mean_rew_batch, val_loss_batch, 
                                    loss.item(), entropy_loss.item(), 
                                    optimizer.param_groups[0]["lr"]])

        if global_step % config.CHECKPOINT_FREQ == 0:
            if not os.path.exists(config.CHECKPOINT_DIR): os.makedirs(config.CHECKPOINT_DIR)
            path = os.path.join(config.CHECKPOINT_DIR, config.MODEL_NAME)
            torch.save({
                'global_step': global_step, 
                'model_state_dict': agent.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict()
            }, path)
            print(f"Checkpoint saved: {path}")

    envs.close()

if __name__ == "__main__":
    main()