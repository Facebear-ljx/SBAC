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
from Network.Actor_Critic_net import Actor, BC_standard, Q_critic, Alpha, W
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


class SBAC:
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

        super(SBAC, self).__init__()
        self.env = gym.make(env_name)

        num_state = self.env.observation_space.shape[0]
        num_action = self.env.action_space.shape[0]
        self.dataset = self.env.get_dataset()
        self.replay_buffer = ReplayBuffer(state_dim=num_state, action_dim=num_action, device=device)
        self.s_mean, self.s_std = self.replay_buffer.convert_D4RL(d4rl.qlearning_dataset(self.env, self.dataset),
                                                                  scale_rewards=False, scale_state=True)

        # self.env2 = gym.make('hopper-expert-v2')
        # self.dataset2 = self.env2.get_dataset()
        # self.replay_buffer2 = ReplayBuffer(state_dim=num_state, action_dim=num_action, device=device)
        # self.replay_buffer2.convert_D4RL(d4rl.qlearning_dataset(self.env2, self.dataset2), scale_rewards=False, scale_state=False)

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
        self.actor_optim = optim.Adam(self.actor_net.parameters(), self.lr_actor)

        self.bc_standard_net = BC_standard(num_state, num_action, num_hidden, device).float().to(device)
        self.bc_standard_optim = optim.Adam(self.bc_standard_net.parameters(), lr_actor)

        # q_miu
        self.q_net = Q_critic(num_state, num_action, num_hidden, device).float().to(device)
        self.target_q_net = copy.deepcopy(self.q_net)
        self.q_optim = optim.Adam(self.q_net.parameters(), self.lr_critic)

        self.q_pi_net = Q_critic(num_state, num_action, num_hidden, device).float().to(device)
        self.target_q_pi_net = copy.deepcopy(self.q_pi_net)
        self.q_pi_optim = optim.Adam(self.q_pi_net.parameters(), self.lr_critic)

        self.alpha_net = Alpha(num_state, num_action, num_hidden, device).float().to(device)
        self.alpha_optim = optim.Adam(self.alpha_net.parameters(), self.lr_critic)

        self.w_net = W(num_state, num_hidden, device).to(device)
        self.w_optim = optim.Adam(self.w_net.parameters(), self.lr_critic)

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

    def pretrain_bc_standard(self):
        for i in range(self.warmup_steps):
            state, action, _, _, _, _ = self.replay_buffer.sample(self.batch_size)
            bc_loss = self.train_bc_standard(s=state, a=action)
            if i % 5000 == 0:
                miu_reward = self.rollout_evaluate(pi='miu')
                wandb.log({"bc_loss": bc_loss,
                          "miu_reward": miu_reward
                        })

        torch.save(self.bc_standard_net.state_dict(), self.file_loc[1])

    def learn(self, total_time_step=1e+6):
        i_so_far = 0

        # load pretrain model parameters
        if os.path.exists(self.file_loc[1]):
            self.load_parameters()
        else:
            self.pretrain_bc_standard()

        # train !
        while i_so_far < total_time_step:
            i_so_far += 1

            # sample data
            state, action, next_state, _, reward, not_done = self.replay_buffer.sample(self.batch_size)

            # train Q
            q_pi_loss, q_pi_mean = self.train_Q_pi(state, action, next_state, reward, not_done)
            q_miu_loss, q_miu_mean = self.train_Q_miu(state, action, next_state, reward, not_done)

            # Actor_standard, calculate the log(\miu)
            action_pi = self.actor_net.get_action(state)
            log_miu = self.bc_standard_net.get_log_density(state, action_pi)

            A = self.q_net(state, action_pi)

            if self.auto_alpha:
                self.update_alpha(state, action_pi.detach(), log_miu.detach())
                self.alpha = torch.mean(torch.clip(self.get_alpha(state, action_pi), ALPHA_MIN, ALPHA_MAX)).detach()

            # policy update
            actor_loss = torch.mean(-1. * A - self.alpha * log_miu)
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # save model
            if i_so_far % 100000 == 0:
                self.save_parameters()

            # log_summary
            if i_so_far % 3000 == 0:
                pi_reward = self.rollout_evaluate(pi='pi')
                miu_reward = self.rollout_evaluate(pi='miu')

                wandb.log({"actor_loss": actor_loss.item(),
                           "pi_reward": pi_reward,
                           "miu_reward": miu_reward,
                           "q_pi_loss": q_pi_loss,
                           "q_pi_mean": q_pi_mean,
                           "q_miu_loss": q_miu_loss,
                           "q_miu_mean": q_miu_mean,
                           "log_miu": log_miu.mean().item(),
                           "Q_pi-Q_miu": q_pi_mean - q_miu_mean,
                           "reward": reward.mean().item(),
                           "it_steps": i_so_far
                           })

    def online_fine_tune(self, total_time_step=1e+6):
        i_so_far = 0
        t_so_far = 0
        rollout_length = 2048

        # load pretrain model parameters
        if os.path.exists(self.file_loc[1]):
            self.load_q_actor_parameters()
        else:
            self.learn(total_time_step=1e+6)

        # Online fine-tuning !
        while i_so_far < total_time_step:
            # online collect data
            total_rollout_steps = self.rollout_online(rollout_length)
            i_so_far += 1
            t_so_far += total_rollout_steps

            # copy the actor parameters
            for target_param, param in zip(self.actor_last_net.parameters(), self.actor_net.parameters()):
                target_param.data.copy_(param)

            for _ in range(10):
                state, action, next_state, reward, not_done = self.sample_from_online_buffer(total_rollout_steps,
                                                                                             self.batch_size)

                # Q_pi and Q_miu fine-tuning
                q_pi_loss, q_pi_mean = self.train_Q_pi(state, action, next_state, reward, not_done)
                q_miu_loss, q_miu_mean = self.train_Q_miu(state, action, next_state, reward, not_done)

                # Actor_standard, calculate the log(\miu)
                action_pi = self.actor_net.get_action(state)
                state_norm = (state - self.s_mean) / (self.s_std + 1e-5)
                log_pi_last = self.actor_last_net.get_log_density(state, action_pi)
                log_miu = self.bc_standard_net.get_log_density(state_norm, action_pi)

                A = self.q_net(state, action_pi)

                # policy update
                actor_loss = torch.mean(-1. * A - 0.2 * log_miu)
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                if i_so_far % 3000 == 0:
                    pi_reward = self.rollout_evaluate(pi='pi')
                    miu_reward = self.rollout_evaluate(pi='miu')

                    wandb.log({"actor_loss": actor_loss.item(),
                               "pi_reward": pi_reward,
                               "miu_reward": miu_reward,
                               "q_pi_loss": q_pi_loss,
                               "q_pi_mean": q_pi_mean,
                               "q_miu_loss": q_miu_loss,
                               "q_miu_mean": q_miu_mean,
                               "log_miu": log_miu.mean().item(),
                               "log_pi_last": log_pi_last.mean().item(),
                               "Q_pi-Q_miu": q_pi_mean - q_miu_mean,
                               "reward": reward.mean().item()
                               })

    def build_w_loss(self, s1, a1, s2):
        w_s1 = self.w_net(s1)
        w_s2 = self.w_net(s2)
        logp = self.actor_net.get_log_density(s1, a1)
        logb = self.bc_standard_net.get_log_density(s1, a1)
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

    def train_Q_pi(self, s, a, next_s, r, not_done):
        next_s_pi = next_s
        next_action_pi = self.actor_net.get_action(next_s_pi)

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
        self.bc_standard_net.train()
        action_log_prob = -1. * self.bc_standard_net.get_log_density(s, a)
        loss_bc = torch.mean(action_log_prob)

        self.bc_standard_optim.zero_grad()
        loss_bc.backward()
        self.bc_standard_optim.step()
        return loss_bc.cpu().detach().numpy().item()

    def train_Q_miu(self, s, a, next_s, r, not_done):
        next_s_miu = next_s - self.s_mean
        next_s_miu /= (self.s_std + 1e-5)
        next_action_miu = self.bc_standard_net.get_action(next_s_miu)

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

    def rollout_evaluate(self, pi='pi'):
        ep_rews = 0.  # episode reward
        self.bc_standard_net.eval()
        # self.actor_net.eval()
        # rollout one time
        for _ in range(1):
            ep_rews = 0.
            state = self.env.reset()
            ep_t = 0  # episode length
            while True:
                ep_t += 1
                if pi == 'pi':
                    action = self.actor_net.get_action(state).cpu().detach().numpy()
                else:
                    state = (state - self.s_mean) / (self.s_std + 1e-5)
                    action = self.bc_standard_net.get_action(np.expand_dims(state, 0)).cpu().detach().numpy()
                state, reward, done, _ = self.env.step(action)
                ep_rews += reward
                if done:
                    break

        return ep_rews

    def distance_function(self):
        self.load_q_actor_parameters()
        state, action, _, _, reward, _ = self.replay_buffer.sample(self.batch_size)
        reward = reward.cpu().detach().numpy().squeeze()
        q_pi = self.q_pi_net(state, action).cpu().detach().numpy().squeeze()
        q_miu = self.q_net(state, action).cpu().detach().numpy().squeeze()
        plt.figure(figsize=(5, 5), dpi=100)
        plt.subplot(2, 3, 1)
        plt.scatter(reward, q_pi)
        plt.xlabel("reward")
        plt.ylabel("q_pi")
        plt.subplot(2, 3, 2)
        plt.scatter(reward, q_miu)
        plt.xlabel("reward")
        plt.ylabel("q_miu")
        plt.subplot(2, 3, 3)
        plt.scatter(reward, q_pi - q_miu)
        plt.xlabel("reward")
        plt.ylabel("q_pi-q_miu")

        state2, action2, _, _, reward2, _ = self.replay_buffer2.sample(self.batch_size)
        reward2 = reward2.cpu().detach().numpy().squeeze()
        q_pi2 = self.q_pi_net(state2, action2).cpu().detach().numpy().squeeze()
        q_miu2 = self.q_net(state2, action2).cpu().detach().numpy().squeeze()
        plt.subplot(2, 3, 4)
        plt.scatter(reward2, q_pi2, edgecolors='r')
        plt.xlabel("reward")
        plt.ylabel("q_pi")
        plt.subplot(2, 3, 5)
        plt.scatter(reward2, q_miu2, edgecolors='r')
        plt.xlabel("reward")
        plt.ylabel("q_miu")
        plt.subplot(2, 3, 6)
        plt.scatter(reward2, q_pi2 - q_miu2, edgecolors='r')
        plt.xlabel("reward")
        plt.ylabel("q_pi-q_miu")
        plt.show()

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
        self.q_pi_net.load_state_dict(torch.load(self.file_loc[2]))
        self.target_q_pi_net.load_state_dict(torch.load(self.file_loc[2]))
        self.actor_net.load_state_dict(torch.load(self.file_loc[3]))
