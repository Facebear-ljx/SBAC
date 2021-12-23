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
from Network.Actor_Critic_net import Actor, BC_standard, Q_critic, Alpha, W, V_critic
import matplotlib.pyplot as plt

ALPHA_MAX = 500.
ALPHA_MIN = 0.2
EPS = 1e-8


class TEST:
    max_grad_norm = 0.5
    soft_update = 0.005

    def __init__(self, env_name,
                 num_hidden,
                 Use_W=True,
                 lr_actor=1e-5,
                 lr_critic=1e-3,
                 gamma=0.99,
                 warmup_steps=1000000,
                 alpha=100,
                 auto_alpha=False,
                 epsilon=1,
                 batch_size=256,
                 device='cpu'):

        super(TEST, self).__init__()
        self.env = gym.make(env_name)

        num_state = self.env.observation_space.shape[0]
        num_action = self.env.action_space.shape[0]
        self.dataset = self.env.get_dataset()
        self.replay_buffer = ReplayBuffer(state_dim=num_state, action_dim=num_action, device=device)
        self.s_mean, self.s_std = self.replay_buffer.convert_D4RL(d4rl.qlearning_dataset(self.env, self.dataset),
                                                                  scale_rewards=True, scale_state=True)

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.auto_alpha = auto_alpha
        self.alpha = alpha
        self.epsilon = epsilon
        self.s_buffer, self.a_buffer, self.next_s_buffer, self.not_done_buffer, self.r_buffer = [], [], [], [], []

        self.actor_net = Actor(num_state, num_action, num_hidden, device).float().to(device)
        self.actor_last_net = Actor(num_state, num_action, num_hidden, device).float().to(device)
        self.bc_standard_net = BC_standard(num_state, num_action, num_hidden, device).float().to(device)
        self.q_net = Q_critic(num_state, num_action, num_hidden, device).float().to(device)
        self.q_pi_net = Q_critic(num_state, num_action, num_hidden, device).float().to(device)
        self.target_q_net = Q_critic(num_state, num_action, num_hidden, device).float().to(device)
        self.target_q_pi_net = Q_critic(num_state, num_action, num_hidden, device).float().to(device)
        self.alpha_net = Alpha(num_state, num_action, num_hidden, device).float().to(device)
        self.w_net = W(num_state, num_hidden, device).to(device)
        self.v_net = V_critic(num_state, num_hidden, device).to(device)
        self.v_target = copy.deepcopy(self.v_net)

        self.actor_optim = optim.Adam(self.actor_net.parameters(), self.lr_actor)
        self.bc_standard_optim = optim.Adam(self.bc_standard_net.parameters(), 1e-3)
        self.q_optim = optim.Adam(self.q_net.parameters(), self.lr_critic)
        self.q_pi_optim = optim.Adam(self.q_pi_net.parameters(), self.lr_critic)
        self.alpha_optim = optim.Adam(self.alpha_net.parameters(), self.lr_critic)
        self.w_optim = optim.Adam(self.w_net.parameters(), self.lr_critic)
        self.v_optim = optim.Adam(self.v_net.parameters(), self.lr_critic)

        self.Use_W = Use_W
        self.file_loc = prepare_env(env_name)

        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,  # time_steps so far
            'i_so_far': 0,  # iterations so far
            'batch_lens': [],  # episodic lengths in batch
            'batch_rews': [],  # episodic returns in batch
        }
        # from statics import reward_and_done

    def learn(self, total_time_step=1e+6):
        i_so_far = 0

        self.load_parameters()
        # train !
        while i_so_far < total_time_step:
            i_so_far += 1

            # sample data
            state, action, next_state, next_action, reward, not_done = self.replay_buffer.sample(self.batch_size)

            # train Q and V
            v_miu_mean = self.train_V_miu(state, next_state, reward, not_done)
            q_pi_loss, q_pi_mean, A_diff = self.train_Q_pi(state, action, next_state, reward, not_done)
            q_miu_loss, q_miu_mean = self.train_Q_miu(state, action, next_state, next_action, reward, not_done)

            # Actor_standard, calculate the log(\miu)
            action_pi = self.actor_net.get_action(state)
            log_miu = self.bc_standard_net.get_log_density(state, action_pi)
            log_pi = self.actor_net.get_log_density(state, action)

            A_pi = self.q_pi_net(state, action_pi)
            lmbda = 2.5 / A_pi.abs().mean().detach()

            if self.auto_alpha:
                self.update_alpha(state, action_pi.detach(), log_miu.detach())
                self.alpha = torch.mean(torch.clip(self.get_alpha(state, action_pi), ALPHA_MIN, ALPHA_MAX)).detach()

            # policy update
            actor_loss = torch.mean(- A_pi) + nn.MSELoss()(action_pi, action)
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # evaluate
            if i_so_far % 500 == 0:
                self.rollout_evaluate(pi='pi')
                self.rollout_evaluate(pi='miu')

            wandb.log({"actor_loss": actor_loss.item(),
                       "alpha": self.alpha.item(),
                       "q_pi_loss": q_pi_loss,
                       "q_pi_mean": q_pi_mean,
                       "q_miu_loss": q_miu_loss,
                       "q_miu_mean": q_miu_mean,
                       # "log_miu": log_miu.mean().item(),
                       "log_pi": log_pi.mean().item(),
                       "v_miu_mean": v_miu_mean,
                       "A_diff": A_diff
                       })

    def update_alpha(self, state, action, log_miu):
        alpha = self.alpha_net(state, action)
        self.alpha_optim.zero_grad()
        alpha_loss = torch.mean(alpha * (log_miu + self.epsilon))
        alpha_loss.backward()
        self.alpha_optim.step()

    def get_alpha(self, state, action):
        return self.alpha_net(state, action).detach()

    def train_V_miu(self, s, next_s, r, not_done):
        with torch.no_grad():
            target_v = r + not_done * self.v_target(next_s) * self.gamma
        v_loss = nn.MSELoss()(self.v_net(s), target_v)

        self.v_optim.zero_grad()
        v_loss.backward()
        self.v_optim.step()

        v_mean = self.v_net(s).mean().detach().item()

        for target_param, param in zip(self.v_target.parameters(), self.v_net.parameters()):
            target_param.data.copy_(self.soft_update * param.data + (1 - self.soft_update) * target_param.data)
        return v_mean

    def train_Q_pi(self, s, a, next_s, r, not_done):
        Q_mean = 0.
        with torch.no_grad():
            # MC calculate V_pi
            for _ in range(10):
                next_action_pi = self.actor_net.get_action(next_s)
                Q_mean += self.target_q_pi_net(next_s, next_action_pi)
            Q_mean /= 10.

            # calculate the uncertainty
            next_action_pi = self.actor_net.get_action(next_s)
            Q_pi = self.target_q_pi_net(next_s, next_action_pi)
            Q_miu = self.target_q_net(next_s, next_action_pi)
            A_pi = Q_pi - Q_mean
            A_miu = Q_miu - self.v_target(next_s)
            uncertainty = 0.2 * (Q_pi - Q_miu)
            uncertainty = 0
            # target_q = r + not_done * (0.5 * Q_pi + 0.5 * Q_miu) * self.gamma
            target_q = r + not_done * (self.target_q_pi_net(next_s, next_action_pi) - uncertainty) * self.gamma

        loss_q = nn.MSELoss()(target_q, self.q_pi_net(s, a))

        self.q_pi_optim.zero_grad()
        loss_q.backward()
        self.q_pi_optim.step()
        q_mean = self.q_pi_net(s, a).mean().item()
        A_diff = (A_pi - A_miu).mean().detach().item()

        # target Q update
        for target_param, param in zip(self.target_q_pi_net.parameters(), self.q_pi_net.parameters()):
            target_param.data.copy_(self.soft_update * param.data + (1 - self.soft_update) * target_param.data)

        return loss_q, q_mean, A_diff

    def train_Q_miu(self, s, a, next_s, next_a, r, not_done):
        target_q = r + not_done * self.target_q_net(next_s, next_a).detach() * self.gamma
        loss_q = nn.MSELoss()(target_q, self.q_net(s, a))

        self.q_optim.zero_grad()
        loss_q.backward()
        self.q_optim.step()
        q_mean = self.q_net(s, a).mean().item()

        # target Q update
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.soft_update * param.data + (1 - self.soft_update) * target_param.data)
        return loss_q, q_mean

    def rollout_evaluate(self, pi='pi'):
        ep_rews = 0.  # episode reward
        self.bc_standard_net.eval()
        # rollout one time
        for _ in range(1):
            ep_rews = 0.
            state = self.env.reset()
            ep_t = 0  # episode length
            while True:
                ep_t += 1
                action = self.actor_net.get_action(state).cpu().detach().numpy()
                state, reward, done, _ = self.env.step(action)
                ep_rews += reward
                if done:
                    break

        wandb.log({"pi_episode_reward": ep_rews})

    def save_parameters(self):
        torch.save(self.q_net.state_dict(), self.file_loc[0])
        torch.save(self.q_pi_net.state_dict(), self.file_loc[2])
        torch.save(self.actor_net.state_dict(), self.file_loc[3])
        # torch.save(self.target_q_net.state_dict(), self.file_loc[])
        # torch.save(self.target_q_pi_net.state_dict(), self.file_loc[])

    def rollout_online(self, rollout_length):
        # rollout one time
        s_buffer = []
        a_buffer = []
        next_s_buffer = []
        r_buffer = []
        not_done_buffer = []

        total_rollout_steps = 0
        while True:
            state = self.env.reset()
            while True:
                total_rollout_steps += 1
                action = self.actor_net.get_action(state).cpu().detach().numpy()
                next_state, reward, done, _ = self.env.step(action)

                s_buffer.append(state)
                a_buffer.append(action)
                next_s_buffer.append(next_state)
                r_buffer.append(reward)
                not_done_buffer.append(1 - done)

                state = next_state
                if done:
                    break
            if total_rollout_steps >= rollout_length:
                break

        self.s_buffer = np.array(s_buffer)
        self.a_buffer = np.array(a_buffer)
        self.next_s_buffer = np.array(next_s_buffer)
        self.r_buffer = np.expand_dims(np.array(r_buffer), axis=1)
        self.not_done_buffer = np.expand_dims(np.array(not_done_buffer), axis=1)
        return total_rollout_steps

    def sample_from_online_buffer(self, total_rollout_steps, batch_size):
        ind = np.random.randint(0, total_rollout_steps, size=batch_size)
        return (
            torch.FloatTensor(self.s_buffer[ind]).to(self.device),
            torch.FloatTensor(self.a_buffer[ind]).to(self.device),
            torch.FloatTensor(self.next_s_buffer[ind]).to(self.device),
            torch.FloatTensor(self.r_buffer[ind]).to(self.device),
            torch.FloatTensor(self.not_done_buffer[ind]).to(self.device)
        )

    def load_parameters(self):
        self.bc_standard_net.load_state_dict(torch.load(self.file_loc[1]))

    def load_q_actor_parameters(self):
        self.bc_standard_net.load_state_dict(torch.load(self.file_loc[1]))
        self.q_net.load_state_dict(torch.load(self.file_loc[0]))
        self.target_q_net.load_state_dict(torch.load(self.file_loc[0]))
