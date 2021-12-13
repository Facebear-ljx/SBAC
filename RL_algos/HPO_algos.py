import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import wandb
import time
import gym
import d4rl
import os
from torch.distributions import Normal
import torch.nn.functional as F
from Network.Actor_Critic_net import Actor, BC_standard, Q_critic, Alpha, W, BC_VAE

ALPHA_MAX = 500
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


class HPO:
    max_grad_norm = 0.5
    soft_update = 0.005

    def __init__(self, env, num_state, num_action, num_hidden, replay_buffer, file_loc, Use_W, HPO_=False,
                 lr_actor=1e-5, lr_critic=1e-3, gamma=0.99, warmup_steps=30000, alpha=100,
                 auto_alpha=False, epsilon=1, batch_size=256, device='cpu', A_norm=False):
        super(HPO, self).__init__()
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.auto_alpha = auto_alpha
        self.alpha = alpha
        self.epsilon = epsilon
        self.cov_var = torch.full(size=(3,), fill_value=0.5).to(device)
        self.cov_mat = torch.diag(self.cov_var).to(device)

        self.actor_net = Actor(num_state, num_action, num_hidden, device).float().to(device)
        self.bc_standard_net = BC_standard(num_state, num_action, num_hidden, device).float().to(device)
        self.q_net = Q_critic(num_state, num_action, num_hidden, device).float().to(device)
        self.q_pi_net = Q_critic(num_state, num_action, num_hidden, device).float().to(device)
        self.target_q_net = Q_critic(num_state, num_action, num_hidden, device).float().to(device)
        self.target_q_pi_net = Q_critic(num_state, num_action, num_hidden, device).float().to(device)
        self.alpha_net = Alpha(num_state, num_action, num_hidden, device).float().to(device)
        self.w_net = W(num_state, num_hidden, device).to(device)
        self.bc_vae = BC_VAE(num_state, num_action, num_hidden, 4, device).to(device)

        self.actor_optim = optim.Adam(self.actor_net.parameters(), self.lr_actor, weight_decay=0.01)
        self.bc_standard_optim = optim.Adam(self.bc_standard_net.parameters(), lr_actor, weight_decay=0.01)
        self.q_optim = optim.Adam(self.q_net.parameters(), self.lr_critic, weight_decay=0.01)
        self.q_pi_optim = optim.Adam(self.q_pi_net.parameters(), self.lr_critic, weight_decay=0.01)
        self.alpha_optim = optim.Adam(self.alpha_net.parameters(), self.lr_critic)
        self.w_optim = optim.Adam(self.w_net.parameters(), self.lr_critic, weight_decay=0.01)
        self.bc_vae_optim = optim.Adam(self.bc_vae.parameters(), self.lr_actor, weight_decay=0.01)

        self.env = env
        self.replay_buffer = replay_buffer
        self.A_norm = A_norm
        self.Use_W = Use_W
        self.HPO_ = HPO_
        # network paremeters location
        self.file_loc = file_loc
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,  # time_steps so far
            'i_so_far': 0,  # iterations so far
            'batch_lens': [],  # episodic lengths in batch
            'batch_rews': [],  # episodic returns in batch
        }

    def pretrain_bc_standard(self):
        for i in range(self.warmup_steps):
            state, action, _, _, _, _ = self.replay_buffer.sample(2048)
            self.train_bc_standard(s=state, a=action)
            if i % 500 == 0:
                self.rollout(pi='miu')
        torch.save(self.bc_standard_net.state_dict(), self.file_loc[1])

    # def pretrain_bc_vae(self):
    #     for _ in range(self.warmup_steps):
    #         state, action, _, _, _, _ = self.replay_buffer.sample(2048)
    #         self.train_bc_vae(state, action)
    #     torch.save(self.bc_vae.state_dict(), self.bc_vae_file_loc)

    # def pretrain_Q(self):
    #     for _ in range(self.warmup_steps):
    #         state, action, next_state, next_action, reward, not_done = self.replay_buffer.sample(2048)
    #         self.train_Q_miu(s=state, a=action, next_s=next_state, r=reward, not_done=not_done)
    #     torch.save(self.q_net.state_dict(), self.q_file_loc)

    def learn_SBAC_plus(self, total_time_step=1e+10):
        i_so_far = 0

        # load pretrain model parameters
        self.load_parameters()

        # train !
        while i_so_far < total_time_step:
            i_so_far += 1

            # sample data
            state, action, next_state, _, reward, not_done = self.replay_buffer.sample(self.batch_size)

            # train Q
            q_pi_loss, q_pi_mean = self.train_Q_pi(state, action, next_state, reward, not_done)
            q_miu_loss, q_miu_mean = self.train_Q_miu(state, action, next_state, reward, not_done)

            # Actor_standard, calculate the log(\pi)
            action_pi = self.actor_net.get_action(state)
            log_pi = self.actor_net.get_log_density(state, action_pi)

            log_miu = self.bc_standard_net.get_log_density(state, action_pi)
            log_pi_from_data = self.actor_net.get_log_density(state, action)

            # importance sampling ratio W
            w_loss = self.build_w_loss(state, action, next_state)
            self.w_optim.zero_grad()
            w_loss.backward()
            self.w_optim.step()
            w = self.w_net(state)

            A = self.q_net(state, action_pi)
            # norm A
            # if self.A_norm:
            #     A = (A - A.mean()) / (A.std() + 1e-10)

            if self.auto_alpha:
                self.update_alpha(state, action_pi.detach(), log_miu.detach())
                self.alpha = torch.mean(torch.clip(self.get_alpha(state, action_pi), max=ALPHA_MAX)).detach()

            # pro_pi = torch.clip(torch.exp(log_pi), min=1e-8, max=1e+8)
            # pro_miu = torch.clip(torch.exp(log_miu), min=1e-8, max=1e+8)
            # if self.Use_W:
            #     if self.HPO_:
            #         actor_loss = torch.mean(-1. * A.detach() * w * 15 * (pro_pi - pro_miu) - log_pi_from_data)  # HPO
            #     else:
            #         actor_loss = torch.mean(-1. * A * w * 15 - log_pi_from_data)  # SBAC
            # else:
            #     if self.HPO_:
            #         actor_loss = torch.mean(-1. * A.detach() * (pro_pi - pro_miu) - log_pi_from_data)
            #     else:
            actor_loss = torch.mean(-1.*A - 1./w.detach() * self.alpha * log_miu)
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            if i_so_far % 500 == 0:
                self.rollout(pi='pi')
                self.rollout(pi='miu')

            # if i_so_far % 100 == 0:
            wandb.log({"actor_loss": actor_loss.item(),
                       "alpha": self.alpha.item(),
                       # "w_loss": w_loss.item(),
                       "q_pi_loss": q_pi_loss,
                       "q_pi_mean": q_pi_mean,
                       "q_miu_loss": q_miu_loss,
                       "q_miu_mean": q_miu_mean,
                       "log_miu": log_miu.mean().item(),
                       # "log_pi": log_pi.mean().item(),
                       # "log_miu_from_data": log_miu_from_data.mean().item(),
                       # "log_pi_from_data": log_miu_from_data.mean().item()
                       })

    def build_w_loss(self, s1, a1, s2):
        w_s1 = self.w_net(s1)
        w_s2 = self.w_net(s2)
        logp = self.evaluate_pi(s1, a1)
        logb = self.evaluate_bc_standard(s1, a1)
        ratio = torch.clip(logp - logb, min=-10., max=10.)

        k11 = torch.mean(torch.mean(laplacian_kernel(s2, s2) * w_s2, dim=0) * w_s2, dim=1)
        k12 = torch.mean(torch.mean(laplacian_kernel(s2, s1) * w_s2, dim=0) * (
                1 - self.gamma + self.gamma * torch.exp(ratio) * w_s1), dim=1)
        k22 = torch.mean(
            torch.mean(laplacian_kernel(s1, s1) * (1 - self.gamma + self.gamma * torch.exp(ratio) * w_s1), dim=0) * (
                    1 - self.gamma + self.gamma * torch.exp(ratio) * w_s1), dim=1)

        w_loss = torch.mean(k11 - 2 * k12 + k22)
        return w_loss

    def update_alpha(self, state, action, log_miu):
        alpha = self.alpha_net(state, action)
        self.alpha_optim.zero_grad()
        alpha_loss = torch.mean(alpha * (log_miu + self.epsilon))
        alpha_loss.backward()
        self.alpha_optim.step()

    def get_alpha(self, state, action):
        return self.alpha_net(state, action).detach()

    # def train_bc_vae(self, state, action):
    #     action_mean, action_sigma, mean, sigma = self.bc_vae(state, action)
    #     distribution = Normal(action_mean, action_sigma)
    #     recon_loss = torch.sum(-1 * distribution.log_prob(action))
    #
    #     kl_loss = - 0.5 * torch.sum(1 + sigma - mean.pow(2) - sigma.exp())
    #     loss = recon_loss + kl_loss
    #     self.bc_vae_optim.zero_grad()
    #     loss.backward()
    #     self.bc_vae_optim.step()
    #     return loss.item()

    def train_Q_pi(self, s, a, next_s, r, not_done):
        # log_prob = self.actor_net.get_log_density(s, a)
        next_action_pi = self.actor_net.get_action(next_s)

        target_q = r + not_done * self.target_q_pi_net(next_s, next_action_pi).detach() * self.gamma
        loss_q = nn.MSELoss()(target_q, self.q_pi_net(s, a))

        self.q_pi_optim.zero_grad()
        loss_q.backward()
        self.q_pi_optim.step()
        q_mean = self.q_pi_net(s, a).mean().item()

        # target Q update
        for target_param, param in zip(self.target_q_pi_net.parameters(), self.q_pi_net.parameters()):
            target_param.data.copy_(self.soft_update * param + (1 - self.soft_update) * target_param)

        return loss_q, q_mean

    def train_bc_standard(self, s, a):
        # action = self.bc_standard_net.get_action(s)
        # loss_bc = nn.MSELoss()(action, a)
        action_log_prob = -1. * self.bc_standard_net.get_log_density(s, a)
        loss_bc = torch.mean(action_log_prob)

        self.bc_standard_optim.zero_grad()
        loss_bc.backward()
        # torch.nn.utils.clip_grad_norm_(self.bc_standard_net.parameters(), self.max_grad_norm)
        self.bc_standard_optim.step()
        wandb.log({"bc_standard_loss": loss_bc.item()})

    def train_Q_miu(self, s, a, next_s, r, not_done):
        next_action_miu = self.bc_standard_net.get_action(next_s)

        target_q = r + not_done * self.target_q_net(next_s, next_action_miu).detach() * self.gamma
        loss_q = nn.MSELoss()(target_q, self.q_net(s, a))

        self.q_optim.zero_grad()
        loss_q.backward()
        self.q_optim.step()
        q_mean = self.q_net(s, a).mean().item()

        # target Q update
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.soft_update * param + (1 - self.soft_update) * target_param)
        return loss_q, q_mean

    def choose_action_from_pi(self, s):
        action = self.actor_net.get_action(s)
        return action.cpu().detach().numpy()

    def choose_action_from_miu(self, s):
        action = self.bc_standard_net.get_action(s)
        return action.cpu().detach().numpy()

    # def choose_action_from_vae(self, s):
    #     z = torch.randn(4)
    #     z = torch.tensor(z, dtype=torch.float).unsqueeze(0).to(self.device)
    #     s = torch.tensor(s, dtype=torch.float).unsqueeze(0).to(self.device)
    #     action_mean, action_sigma = self.bc_vae.decode(z, s)
    #     distribution = Normal(action_mean, action_sigma)
    #     action = distribution.rsample()
    #     return action.cpu().detach().numpy()

    def evaluate_pi(self, s, a):
        log_prob = self.actor_net.get_log_density(s, a)
        return log_prob

    def evaluate_bc_standard(self, s, a):
        log_prob = self.bc_standard_net.get_log_density(s, a)
        return log_prob

    # def evaluate_bc_vae(self, s, a_pi):
    #     z = torch.randn((self.batch_size, 4)).to(self.device)
    #     s = torch.tensor(s, dtype=torch.float).to(self.device)
    #     action_mean, action_sigma = self.bc_vae.decode(z, s)
    #     distribution_miu = Normal(action_mean, action_sigma)
    #     log_miu = torch.mean(distribution_miu.log_prob(a_pi), dim=1).unsqueeze(1)
    #     return log_miu

    def rollout(self, pi='pi'):
        for _ in range(1):
            ep_rews = 0.
            state = self.env.reset()
            ep_t = 0
            while True:
                # if pi == 'pi':
                # self.env.render()
                ep_t += 1
                if pi == 'pi':
                    action = self.choose_action_from_pi(state)
                else:
                    action = self.choose_action_from_miu(state)
                state, reward, done, _ = self.env.step(action)
                ep_rews += reward

                # If the environment tells us the episode is terminated, break
                if done:
                    break
        # Reshape data as tensors in the shape specified in function description, before returning# ALG STEP 4

        if pi == 'pi':
            wandb.log({"pi_episode_reward": ep_rews})
        else:
            wandb.log({"miu_episode_reward": ep_rews})

    def save_parameters(self):
        torch.save(self.q_net.state_dict(), self.file_loc[0])

    def load_parameters(self):
        # self.q_net.load_state_dict(torch.load(self.q_file_loc))
        self.bc_standard_net.load_state_dict(torch.load(self.file_loc[1]))
        # self.bc_vae.load_state_dict(torch.load(self.file_loc[2]))
