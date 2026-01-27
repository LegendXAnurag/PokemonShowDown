"""
Training script for Pokemon Showdown with MAPPO/IPPO
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
import argparse
from tqdm import tqdm
import os
import re

from environment import PokemonShowdownEnv
from networks import ActorCritic
from utils import (
    ExperienceBuffer, compute_gae, normalize_advantages,
    Logger, save_checkpoint, load_checkpoint,
    get_linear_schedule, RewardTracker, set_seed
)
from config import TRAINING_CONFIG, ENV_CONFIG, RENDER_CONFIG


class MAPPOTrainer:
    """Multi-Agent PPO Trainer"""

    def __init__(self, env, device='cpu', use_shared_critic=False, render_mode=None):
        self.env = env
        self.device = device
        self.use_shared_critic = use_shared_critic
        self.render_mode = render_mode

        sample_agent = env.possible_agents[0]
        obs_dim = env.observation_space(sample_agent).shape[0]

        self.models = {}
        self.optimizers = {}

        for agent in env.possible_agents:
            model = ActorCritic(obs_dim).to(device)
            optimizer = optim.Adam(
                model.parameters(),
                lr=TRAINING_CONFIG['learning_rate']
            )
            self.models[agent] = model
            self.optimizers[agent] = optimizer

        self.buffers = {agent: ExperienceBuffer() for agent in env.possible_agents}

        self.gamma = TRAINING_CONFIG['gamma']
        self.gae_lambda = TRAINING_CONFIG['gae_lambda']
        self.clip_epsilon = TRAINING_CONFIG['clip_epsilon']
        self.value_loss_coef = TRAINING_CONFIG['value_loss_coef']
        self.max_grad_norm = TRAINING_CONFIG['max_grad_norm']
        self.num_epochs = TRAINING_CONFIG['num_epochs']
        self.batch_size = TRAINING_CONFIG['batch_size']
        self.num_minibatches = TRAINING_CONFIG['num_minibatches']

        self.entropy_schedule = get_linear_schedule(
            TRAINING_CONFIG['entropy_coef_start'],
            TRAINING_CONFIG['entropy_coef_end'],
            TRAINING_CONFIG['entropy_decay_steps']
        )

        self.reward_tracker = RewardTracker()
        self.global_step = 0

    def load_from_checkpoint(self, checkpoint_dir, level):
        print(f"Looking for checkpoints in {checkpoint_dir} for level {level}...")

        if not os.path.exists(checkpoint_dir):
            print("Checkpoint directory does not exist, starting fresh.")
            return

        # List all checkpoint files for this level
        pattern = re.compile(f"checkpoint_level_{level}_step_(\\d+).pt")
        files = os.listdir(checkpoint_dir)
        matching_files = [(f, int(pattern.match(f).group(1))) 
                          for f in files if pattern.match(f)]

        if not matching_files:
            print("No checkpoint found, starting fresh.")
            return

        # Pick the checkpoint with the highest step
        latest_file, latest_step = max(matching_files, key=lambda x: x[1])
        filepath = os.path.join(checkpoint_dir, latest_file)

        # Load each agent's model and optimizer
        for agent in self.env.possible_agents:
            step, ckpt_level = load_checkpoint(
                self.models[agent],
                self.optimizers[agent],
                filepath
            )
            print(f"Loaded {agent} from {filepath} (step {step})")

        self.global_step = latest_step
        print(f"Resumed training from global step {self.global_step}")

    def collect_rollout(self, num_steps):
        observations, _ = self.env.reset()

        for step_idx in range(num_steps):
            actions = {}

            for agent in self.env.agents:
                obs_tensor = torch.FloatTensor(observations[agent]).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, log_prob, _, value = self.models[agent].get_action_and_value(obs_tensor)

                actions[agent] = action.item()
                self.buffers[agent].store(
                    observations[agent],
                    action.item(),
                    0,
                    False,
                    value.item(),
                    log_prob.item()
                )

            next_obs, rewards, terms, truncs, _ = self.env.step(actions)

            # Render if in visualization mode
            if self.render_mode == 'human':
                self.env.render()

            for agent in self.env.agents:
                self.buffers[agent].rewards[-1] = rewards.get(agent, 0)
                self.buffers[agent].dones[-1] = (
                    terms.get(agent, False) or truncs.get(agent, False)
                )
                self.reward_tracker.update(agent, rewards.get(agent, 0))

            if all(terms.values()) or all(truncs.values()):
                self.reward_tracker.finish_episode()
                observations, _ = self.env.reset()
            else:
                observations = next_obs

            self.global_step += 1

    def update_policy(self):
        total_losses = defaultdict(list)

        for agent in self.env.possible_agents:
            if len(self.buffers[agent]) == 0:
                continue

            batch = self.buffers[agent].get()

            obs = batch['observations'].to(self.device)
            actions = batch['actions'].to(self.device)
            old_log_probs = batch['log_probs'].to(self.device)
            rewards = batch['rewards'].to(self.device)
            dones = batch['dones'].to(self.device)
            old_values = batch['values'].to(self.device)

            with torch.no_grad():
                _, next_value = self.models[agent].forward(obs[-1:])

            advantages, returns = compute_gae(
                rewards,
                old_values,
                dones,
                next_value.squeeze(),
                self.gamma,
                self.gae_lambda
            )
            advantages = normalize_advantages(advantages)

            for _ in range(self.num_epochs):
                indices = np.random.permutation(len(obs))
                minibatch_size = len(obs) // self.num_minibatches

                for start in range(0, len(obs), minibatch_size):
                    mb_idx = indices[start:start + minibatch_size]
                    if len(mb_idx) == 0:
                        continue

                    mb_obs = obs[mb_idx]
                    mb_actions = actions[mb_idx]
                    mb_old_log_probs = old_log_probs[mb_idx]
                    mb_adv = advantages[mb_idx]
                    mb_ret = returns[mb_idx]

                    _, new_log_probs, entropy, new_values = (
                        self.models[agent].get_action_and_value(mb_obs, mb_actions)
                    )

                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(
                        ratio,
                        1 - self.clip_epsilon,
                        1 + self.clip_epsilon
                    ) * mb_adv

                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = 0.5 * (new_values.squeeze() - mb_ret).pow(2).mean()
                    entropy_loss = -entropy.mean()

                    entropy_coef = self.entropy_schedule(self.global_step)
                    loss = (
                        policy_loss
                        + self.value_loss_coef * value_loss
                        + entropy_coef * entropy_loss
                    )

                    self.optimizers[agent].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.models[agent].parameters(),
                        self.max_grad_norm
                    )
                    self.optimizers[agent].step()

                    total_losses['policy_loss'].append(policy_loss.item())
                    total_losses['value_loss'].append(value_loss.item())
                    total_losses['entropy'].append(entropy.mean().item())
                    total_losses['total_loss'].append(loss.item())

            self.buffers[agent].clear()

        return {k: np.mean(v) for k, v in total_losses.items()}

    def train(self, total_timesteps, log_interval, save_interval, save_dir, log_dir):
        logger = Logger(log_dir)

        steps_per_update = self.batch_size
        num_updates = total_timesteps // steps_per_update

        print(f"Starting training for {total_timesteps} timesteps ({num_updates} updates)")
        print(f"Environment: {self.env.level}")
        print(f"Number of agents: {len(self.env.possible_agents)}")
        if self.render_mode:
            print(f"Render mode: {self.render_mode}")

        log_every = max(1, log_interval // steps_per_update)
        save_every = max(1, save_interval // steps_per_update)

        progress_bar = tqdm(
            range(num_updates),
            desc="Training",
            unit="update",
            dynamic_ncols=True
        )

        for update in progress_bar:
            self.collect_rollout(steps_per_update)
            loss_metrics = self.update_policy()

            if update % log_every == 0:
                reward_stats = self.reward_tracker.get_stats()

                metrics = {
                    **loss_metrics,
                    'reward_mean': reward_stats['mean'],
                    'reward_std': reward_stats['std'],
                    'reward_min': reward_stats['min'],
                    'reward_max': reward_stats['max'],
                    'global_step': self.global_step,
                    'entropy_coef': self.entropy_schedule(self.global_step)
                }

                logger.log(self.global_step, metrics)
                logger.print_metrics(self.global_step, metrics)

                progress_bar.set_postfix({
                    "steps": self.global_step,
                    "r_mean": f"{reward_stats['mean']:.2f}",
                    "entropy": f"{metrics['entropy_coef']:.4f}"
                })

                self.reward_tracker.reset()

            if update % save_every == 0 and update > 0:
                for agent, model in self.models.items():
                    save_checkpoint(
                        model,
                        self.optimizers[agent],
                        self.global_step,
                        self.env.level,
                        save_dir
                    )

        progress_bar.close()
        print("Training complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=str, default='level_1',
                        choices=[f'level_{i}' for i in range(1, 7)])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: auto, cuda, or cpu')
    parser.add_argument('--render', action='store_true',
                        help='Enable visualization during training (slower)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')

    args = parser.parse_args()

    # Auto-detect device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = 'cpu'
            print("CUDA not available, using CPU")
    else:
        device = args.device
        if device == 'cuda' and not torch.cuda.is_available():
            print("WARNING: CUDA requested but not available, falling back to CPU")
            device = 'cpu'
    
    print(f"Training device: {device}")

    set_seed(args.seed)

    # Use human mode for visualization, None for fast training
    render_mode = 'human' if args.render else None
    env = PokemonShowdownEnv(level=args.level, render_mode=render_mode)
    
    # Adjust renderer FPS based on mode
    if render_mode == 'human' and env.renderer is not None:
        env.renderer.fps = RENDER_CONFIG['fps']  # Slow for visualization
    elif env.renderer is not None:
        env.renderer.fps = RENDER_CONFIG['training_fps']  # Fast for training

    trainer = MAPPOTrainer(env, device=device, render_mode=render_mode)

    if args.resume:
        trainer.load_from_checkpoint(args.save_dir, args.level)

    trainer.train(
        total_timesteps=TRAINING_CONFIG['total_timesteps'],
        log_interval=TRAINING_CONFIG['log_interval'],
        save_interval=TRAINING_CONFIG['save_interval'],
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )

    env.close()


if __name__ == '__main__':
    main()