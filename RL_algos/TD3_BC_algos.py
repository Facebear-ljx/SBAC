import copy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import time
import gym
import os
import d4rl
from Sample_Dataset.Sample_from_dataset import ReplayBuffer
from Sample_Dataset.Prepare_env import prepare_env
from Network.Actor_Critic_net import Actor_deterministic, Double_Critic
import matplotlib.pyplot as plt

ALPHA_MAX = 500.
ALPHA_MIN = 0.2
EPS = 1e-8


def laplacian_kernel(x1, x2, sigma=20.0):
    d12 = torch.sum(torch.abs(x1[None] - x2[:, None]), dim=-1)
    k12 = torch.exp(- d12 / sigma)
    return k12


def mmd(x1, x2, kernel, use_sqrt=False):
    k11 = torch.mean(kernel(x1, x1), dim=[0, 1])
    k12 = torch.mean(kernel(x1, x2), dim=[0, 1])
    k22 = torch.mean(kernel(x2, x2), dim=[0, 1])
    if use_sqrt:
        return torch.sqrt(k11 + k22 - 2 * k12 + EPS)
    else:
        return k11 + k22 - 2 * k12


class TD3_BC:
    def __init__(self,
                 env_name,
                 num_hidden=256,
                 gamma=0.99,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2,
                 alpha=2.5,
                 device='cpu'):

        super(TD3_BC, self).__init__()
        # prepare the environment
        self.env = gym.make(env_name)
        num_state = self.env.observation_space.shape[0]
        num_action = self.env.action_space.shape[0]
        self.dataset = self.env.get_dataset()
        self.replay_buffer = ReplayBuffer(state_dim=num_state, action_dim=num_action, device=device)
        self.s_mean, self.s_std = self.replay_buffer.convert_D4RL(d4rl.qlearning_dataset(self.env, self.dataset),
                                                                  scale_rewards=True, scale_state=True)

        # self.env2 = gym.make('hopper-expert-v2')
        # self.dataset2 = self.env2.get_dataset()
        # self.replay_buffer2 = ReplayBuffer(state_dim=num_state, action_dim=num_action, device=device)
        # self.replay_buffer2.convert_D4RL(d4rl.qlearning_dataset(self.env2, self.dataset2), scale_rewards=True,
        #                                  scale_state=True)

        # prepare the actor and critic
        self.actor_net = Actor_deterministic(num_state, num_action, num_hidden, device).float().to(device)
        self.actor_target = copy.deepcopy(self.actor_net)
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=3e-4)

        self.critic_net = Double_Critic(num_state, num_action, num_hidden, device).float().to(device)
        self.critic_target = copy.deepcopy(self.critic_net)
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr=3e-4)

        # hyper-parameters
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.policy_freq = policy_freq
        self.evaluate_freq = 500
        self.noise_clip = noise_clip
        self.alpha = alpha
        self.batch_size = 256
        self.device = device
        self.max_action = 1.

        self.total_it = 0

        # Q and Critic file location
        self.file_loc = prepare_env(env_name)

    def learn(self, total_time_step=1e+6):
        while self.total_it <= total_time_step:
            self.total_it += 1

            # sample data
            state, action, next_state, _, reward, not_done = self.replay_buffer.sample(self.batch_size)

            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

                next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + not_done * self.gamma * target_Q

            Q1, Q2 = self.critic_net(state, action)

            # Critic loss
            critic_loss = nn.MSELoss()(Q1, target_Q) + nn.MSELoss()(Q2, target_Q)

            # Optimize Critic
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # delayed policy updata
            if self.total_it % self.policy_freq == 0:

                # Actor loss
                action_pi = self.actor_net(state)
                Q = self.critic_net.Q1(state, action_pi)
                lmbda = self.alpha / Q.abs().mean().detach()

                bc_loss = nn.MSELoss()(action_pi, action)

                actor_loss = -lmbda * Q.mean() + bc_loss

                # Optimize Actor
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                # update the frozen target models
                for param, target_param in zip(self.critic_net.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor_net.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                wandb.log({"actor_loss": actor_loss.item(),
                           "critic_loss": critic_loss.item(),
                           "Q_mean": Q.mean().item(),
                           })

            if self.total_it % self.evaluate_freq == 0:
                self.rollout_evaluate()

    def rollout_evaluate(self):
        ep_rews = 0.
        state = self.env.reset()
        while True:
            state = (state - self.s_mean)/(self.s_std + 1e-5)
            action = self.actor_net(state).cpu().detach().numpy()
            state, reward, done, _ = self.env.step(action)
            ep_rews += reward
            if done:
                break
        wandb.log({"pi_episode_reward": ep_rews})
