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
from Network.Actor_Critic_net import Actor, Double_Critic
from tqdm import tqdm


class CQL:
    def __init__(self,
                 env_name,
                 num_hidden=256,
                 gamma=0.99,
                 tau=0.005,
                 alpha=1.0,
                 start_steps=int(1e+4),
                 seed=0,
                 auto_alpha=True,
                 device='cpu'):
        """
        Facebear's implementation of SAC (Soft Actor Crtic)
        Paper:https://arxiv.org/pdf/1812.05905.pdf

        :param env_name: your environment name
        :param num_hidden: number of the nodes of hidden layer
        :param gamma: discounting factor
        :param tau: soft target network update coefficient
        :param alpha: initial temperature coefficient
        :param start_steps: the number of the start steps to train the policy
        :param seed: random seed
        :param auto_alpha: whether auto update the alpha
        :param device: cuda or cpu
        """
        super(SAC, self).__init__()
        # prepare the environment

        self.env = gym.make(env_name)
        num_state = self.env.observation_space.shape[0]
        num_action = self.env.action_space.shape[0]
        self.replay_buffer = ReplayBuffer(state_dim=num_state, action_dim=num_action, device=device)

        # set seed
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # prepare the actor and critic
        self.actor_net = Actor(num_state, num_action, num_hidden, device).float().to(device)
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=3e-4)

        self.critic_net = Double_Critic(num_state, num_action, num_hidden, device).float().to(device)
        self.critic_target = copy.deepcopy(self.critic_net)
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr=3e-4)

        # hyper-parameters
        self.gamma = gamma
        self.tau = tau
        self.start_steps = start_steps
        self.evaluate_freq = 3000
        self.batch_size = 256
        self.alpha = alpha
        self.target_entropy = -num_action
        self.auto_alpha = True
        self.device = device
        self.max_action = 1.

        if auto_alpha is True:
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * alpha).requires_grad_(True)
            self.log_alpha_optim = torch.optim.Adam([self.log_alpha, ], lr=3e-4)

    def learn(self, total_time_step=1e+6):
        """
        SAC's learning framework
        :param total_time_step: the total iteration times for training BEAR
        :return: None
        """
        state, done, ep_lens = self.env.reset(), False, 0

        for episode_timesteps in tqdm(range(int(total_time_step))):

            # collect data
            if episode_timesteps <= self.start_steps:
                action = self.env.action_space.sample()
            else:
                action = self.select_action(state)

            next_state, reward, done, _ = self.env.step(action)
            ep_lens += 1
            done_bool = float(done) if episode_timesteps < 1000 else 0

            # add data to replay buffer
            self.replay_buffer.add(state, action, next_state, reward, done_bool)
            state = next_state

            if done or (ep_lens >= 1000):
                state, done, ep_lens = self.env.reset(), False, 0

            # train agent after collecting sufficient data
            if episode_timesteps >= self.start_steps:
                # sample data
                s, a, next_s, _, r, not_done = self.replay_buffer.sample(self.batch_size)

                # update Critic
                critic_loss = self.train_Q_pi(s, a, next_s, r, not_done)

                # update Actor and temperature
                Q_mean, actor_loss, alpha_loss = self.train_actor_temperature(s)

                if episode_timesteps % self.evaluate_freq == 0:
                    ep_rews = self.rollout_evaluate()
                    wandb.log({"Q_loss": critic_loss,
                               "Q_mean": Q_mean,
                               "alpha_loss": alpha_loss,
                               "actor_loss": actor_loss,
                               "alpha": self.alpha.cpu().detach().numpy().item(),
                               "steps": episode_timesteps,
                               "episode_rewards": ep_rews
                               })

    def train_Q_pi(self, state, action, next_state, reward, not_done):
        with torch.no_grad():
            next_action, log_next_a, _ = self.actor_net(next_state)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * (target_Q - self.alpha * log_next_a)

        Q1, Q2 = self.critic_net(state, action)

        # Critic loss
        critic_loss = nn.MSELoss()(Q1, target_Q) + nn.MSELoss()(Q2, target_Q)

        # Optimize Critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        return critic_loss.cpu().detach().numpy().item()

    def train_actor_temperature(self, state):
        # Actor loss
        action_pi, log_a, _ = self.actor_net(state)
        Q1, Q2 = self.critic_net(state, action_pi)
        Q = torch.min(Q1, Q2)

        # Entropy regularized policy loss
        actor_loss = (self.alpha * log_a - Q).mean()

        # Optimize Actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # update temperature
        alpha_loss = self.train_alpha(log_a)

        # update the frozen target models
        for param, target_param in zip(self.critic_net.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return Q.mean().cpu().detach().numpy().item(), actor_loss.cpu().detach().numpy().item(), alpha_loss

    def train_alpha(self, log_pi):
        # alpha_loss in equation 18
        alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()

        # alpha optim
        self.log_alpha_optim.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optim.step()

        self.alpha = self.log_alpha.exp().detach()
        return alpha_loss.cpu().detach().numpy().item()

    def select_action(self, state):
        action = self.actor_net.get_action(state)
        return action.cpu().detach().numpy()

    def rollout_evaluate(self):
        ep_rews = 0.
        ep_lens = 0
        state = self.env.reset()
        while True:
            ep_lens += 1
            action = self.select_action(state)
            state, reward, done, _ = self.env.step(action)
            ep_rews += reward
            if done:
                break
        return ep_rews
