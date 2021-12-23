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


class SAC:
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

        super(SAC, self).__init__()
        self.env = gym.make(env_name)

        num_state = self.env.observation_space.shape[0]
        num_action = self.env.action_space.shape[0]
        self.dataset = self.env.get_dataset()
        self.replay_buffer = ReplayBuffer(state_dim=num_state, action_dim=num_action, device=device)
        self.s_mean, self.s_std = self.replay_buffer.convert_D4RL(d4rl.qlearning_dataset(self.env, self.dataset),
                                                                  scale_rewards=False, scale_state=True)

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.auto_alpha = auto_alpha
        self.alpha = alpha
        self.s_buffer, self.a_buffer, self.next_s_buffer, self.not_done_buffer, self.r_buffer = [], [], [], [], []

        self.actor_net = Actor(num_state, num_action, num_hidden, device).float().to(device)
        self.q_net = Q_critic(num_state, num_action, num_hidden, device).float().to(device)
        self.target_q_net = Q_critic(num_state, num_action, num_hidden, device).float().to(device)
        self.alpha_net = Alpha(num_state, num_action, num_hidden, device).float().to(device)

        self.actor_optim = optim.Adam(self.actor_net.parameters(), self.lr_actor)
        self.q_optim = optim.Adam(self.q_net.parameters(), self.lr_critic)
        self.alpha_optim = optim.Adam(self.alpha_net.parameters(), self.lr_critic)

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

        # load pretrain model parameters
        if os.path.exists(self.file_loc[1]):
            self.load_parameters()

        # train !
        while i_so_far < total_time_step:
            i_so_far += 1

            # sample data
            state, action, next_state, _, reward, not_done = self.replay_buffer.sample(self.batch_size)
            state_norm = (state - self.s_mean) / (self.s_std + 1e-5)
            # train Q
            q_pi_loss, q_pi_mean = self.train_Q_pi(state, action, next_state, reward, not_done)
            q_miu_loss, q_miu_mean = self.train_Q_miu(state, action, next_state, reward, not_done)

            # Actor_standard, calculate the log(\miu)
            action_pi = self.actor_net.get_action(state)

            A = self.q_net(state, action_pi)

            # policy update
            actor_loss = torch.mean(-1. * A - self.alpha * log_miu)
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # evaluate
            if i_so_far % 500 == 0:
                self.rollout_evaluate(pi='pi')
                self.rollout_evaluate(pi='miu')

            # save model
            if i_so_far % 100000 == 0:
                self.save_parameters()

            wandb.log({"actor_loss": actor_loss.item(),
                       "alpha": self.alpha.item(),
                       # "w_loss": w_loss.item(),
                       # "w_mean": w.mean().item(),
                       "q_pi_loss": q_pi_loss,
                       "q_pi_mean": q_pi_mean,
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

                # bc fine-tuning
                # self.bc_fine_tune(state, action)

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

                wandb.log({"actor_loss": actor_loss.item(),
                           # "w_loss": w_loss.item(),
                           # "w_mean": w.mean().item(),
                           "q_pi_loss": q_pi_loss,
                           "q_pi_mean": q_pi_mean,
                           "q_miu_loss": q_miu_loss,
                           "q_miu_mean": q_miu_mean,
                           "log_miu": log_miu.mean().item(),
                           "log_pi_last": log_pi_last.mean().item(),
                           "Q_pi-Q_miu": q_pi_mean - q_miu_mean,
                           "reward": reward.mean().item()
                           })

            # evaluate
            if i_so_far % 10 == 0:
                self.rollout_evaluate(pi='pi')
                self.rollout_evaluate(pi='miu')

    def update_alpha(self, state, action, log_miu):
        alpha = self.alpha_net(state, action)
        self.alpha_optim.zero_grad()
        alpha_loss = torch.mean(alpha * (log_miu + self.epsilon))
        alpha_loss.backward()
        self.alpha_optim.step()

    def get_alpha(self, state, action):
        return self.alpha_net(state, action).detach()

    def train_Q_pi(self, s, a, next_s, r, not_done):
        # next_s_pi = next_s - self.s_mean
        # next_s_pi /= (self.s_std + 1e-5)
        next_s_pi = next_s
        next_action_pi = self.actor_net.get_action(next_s_pi)

        target_q = r + not_done * self.target_q_net(next_s, next_action_pi).detach() * self.gamma
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

