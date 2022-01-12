import copy
import random
import time

import torch
import torch.nn as nn
import wandb
import gym
import os
import d4rl
from Sample_Dataset.Sample_from_dataset import ReplayBuffer
from Sample_Dataset.Prepare_env import prepare_env
from Network.Actor_Critic_net import Actor_deterministic, Double_Critic
import numpy as np
from tqdm import tqdm


class Macro:
    def __init__(self,
                 env_name,
                 num_hidden=256,
                 gamma=0.99,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2,
                 alpha=2.5,
                 ratio=10,
                 lmbda=2,
                 seed=0,
                 device='cpu'):
        """
        Facebear's implementation of TD3_BC (A Minimalist Approach to Offline Reinforcement Learning)
        Paper: https://arxiv.org/pdf/2106.06860.pdf

        :param env_name: your gym environment name
        :param num_hidden: the number of the units of the hidden layer of your network
        :param gamma: discounting factor of the cumulated reward
        :param tau: soft update
        :param policy_noise:
        :param noise_clip:
        :param policy_freq: delayed policy update frequency
        :param alpha: the hyper-parameter in equation
        :param ratio:
        :param device:
        """
        super(Macro, self).__init__()
        # prepare the environment
        self.env_name = env_name
        self.env = gym.make(env_name)
        num_state = self.env.observation_space.shape[0]
        num_action = self.env.action_space.shape[0]

        # set seed
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # get dataset 1e+6 samples
        self.dataset = self.env.get_dataset()
        self.replay_buffer = ReplayBuffer(state_dim=num_state, action_dim=num_action, device=device)
        self.dataset = self.replay_buffer.split_dataset(self.env, self.dataset, ratio=ratio)
        self.s_mean, self.s_std = self.replay_buffer.convert_D4RL_macro(self.dataset, scale_rewards=False,
                                                                        scale_state=True, n=lmbda)

        # prepare the actor and critic
        self.actor_net = Actor_deterministic(num_state, num_action, num_hidden, device).float().to(device)
        self.actor_target = copy.deepcopy(self.actor_net)
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=3e-4)

        self.critic_net = Double_Critic(num_state, num_action, num_hidden, 4).float().to(device)
        self.critic_net2 = Double_Critic(num_state, num_action, num_hidden, 4).float().to(device)
        self.critic_net3 = Double_Critic(num_state, num_action, num_hidden, 4).float().to(device)
        self.critic_net4 = Double_Critic(num_state, num_action, num_hidden, 4).float().to(device)
        self.critic_net5 = Double_Critic(num_state, num_action, num_hidden, 4).float().to(device)
        self.critic_net7 = Double_Critic(num_state, num_action, num_hidden, 4).float().to(device)
        self.critic_target = copy.deepcopy(self.critic_net)
        self.critic_target2 = copy.deepcopy(self.critic_net2)
        self.critic_target3 = copy.deepcopy(self.critic_net3)
        self.critic_target4 = copy.deepcopy(self.critic_net4)
        self.critic_target5 = copy.deepcopy(self.critic_net5)
        self.critic_target7 = copy.deepcopy(self.critic_net7)
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr=3e-4)
        self.critic_optim2 = torch.optim.Adam(self.critic_net2.parameters(), lr=3e-4)
        self.critic_optim3 = torch.optim.Adam(self.critic_net3.parameters(), lr=3e-4)
        self.critic_optim4 = torch.optim.Adam(self.critic_net4.parameters(), lr=3e-4)
        self.critic_optim5 = torch.optim.Adam(self.critic_net5.parameters(), lr=3e-4)
        self.critic_optim7 = torch.optim.Adam(self.critic_net7.parameters(), lr=3e-4)

        # hyper-parameters
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.policy_freq = policy_freq
        self.evaluate_freq = 3000
        self.noise_clip = noise_clip
        self.alpha = alpha
        self.batch_size = 256
        self.device = device
        self.max_action = 1.

        self.total_it = 0

        # Q and Critic file location
        # self.file_loc = prepare_env(env_name)

    def learn(self, total_time_step=1e+5):
        """
        TD3_BC's learning framework
        :param total_time_step: the total iteration times for training TD3_BC
        :return: None
        """
        for total_it in tqdm(range(int(total_time_step))):
            total_it += 1

            # sample data
            s0, s1, s2, s3, s4, s5, s6, s7, a0, a1, a2, a3, a4, a5, a6, a7, r0, r1, r2, r3, r4, r5, r6, d7 = self.replay_buffer.sample_multiple(
                self.batch_size)

            # update Critic
            critic_loss_2 = self.train_Q_2(s5, a5, s7, r5, d7)
            critic_loss_3 = self.train_Q_3(s4, a4, s7, r4, d7)
            critic_loss_5 = self.train_Q_5(s2, a2, s7, r2, d7)
            critic_loss_7 = self.train_Q_7(s0, a0, s7, r0, d7)

            # delayed policy update
            if total_it % self.policy_freq == 0:
                actor_loss, bc_loss, Q2_mean, Q3_mean, Q5_mean, Q7_mean = self.train_actor(s0, a0)

                if total_it % self.evaluate_freq == 0:
                    evaluate_reward = self.rollout_evaluate()
                    wandb.log({"actor_loss": actor_loss,
                               "bc_loss": bc_loss,
                               "Q2_loss": critic_loss_2,
                               "Q3_loss": critic_loss_3,
                               "Q5_loss": critic_loss_5,
                               "Q7_loss": critic_loss_7,
                               "Q2_mean": Q2_mean,
                               "Q3_mean": Q3_mean,
                               "Q5_mean": Q5_mean,
                               "Q7_mean": Q7_mean,
                               "evaluate_rewards": evaluate_reward,
                               "it_steps": total_it
                               })

            # if self.total_it % 100000 == 0:
            # self.save_parameters()

    def train_Q_1(self, state, action, next_state, reward, not_done):
        """
        train the Q function of the learned policy: \pi
        """
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * target_Q * self.gamma

        Q1, Q2 = self.critic_net(state, action)

        # Critic loss
        critic_loss = nn.MSELoss()(Q1, target_Q) + nn.MSELoss()(Q2, target_Q)

        # Optimize Critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        return critic_loss.cpu().detach().numpy().item()

    def train_Q_2(self, state, action, next_state, reward, not_done):
        """
        train the Q function of the learned policy: \pi
        """
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target2(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * target_Q * self.gamma

        Q1, Q2 = self.critic_net2(state, action)

        # Critic loss
        critic_loss = nn.MSELoss()(Q1, target_Q) + nn.MSELoss()(Q2, target_Q)

        # Optimize Critic
        self.critic_optim2.zero_grad()
        critic_loss.backward()
        self.critic_optim2.step()
        return critic_loss.cpu().detach().numpy().item()

    def train_Q_3(self, state, action, next_state, reward, not_done):
        """
        train the Q function of the learned policy: \pi
        """
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target3(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * target_Q * self.gamma

        Q1, Q2 = self.critic_net3(state, action)

        # Critic loss
        critic_loss = nn.MSELoss()(Q1, target_Q) + nn.MSELoss()(Q2, target_Q)

        # Optimize Critic
        self.critic_optim3.zero_grad()
        critic_loss.backward()
        self.critic_optim3.step()
        return critic_loss.cpu().detach().numpy().item()

    def train_Q_4(self, state, action, next_state, reward, not_done):
        """
        train the Q function of the learned policy: \pi
        """
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target4(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * target_Q * self.gamma

        Q1, Q2 = self.critic_net4(state, action)

        # Critic loss
        critic_loss = nn.MSELoss()(Q1, target_Q) + nn.MSELoss()(Q2, target_Q)

        # Optimize Critic
        self.critic_optim4.zero_grad()
        critic_loss.backward()
        self.critic_optim4.step()
        return critic_loss.cpu().detach().numpy().item()

    def train_Q_5(self, state, action, next_state, reward, not_done):
        """
        train the Q function of the learned policy: \pi
        """
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target5(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * target_Q * self.gamma

        Q1, Q2 = self.critic_net5(state, action)

        # Critic loss
        critic_loss = nn.MSELoss()(Q1, target_Q) + nn.MSELoss()(Q2, target_Q)

        # Optimize Critic
        self.critic_optim5.zero_grad()
        critic_loss.backward()
        self.critic_optim5.step()
        return critic_loss.cpu().detach().numpy().item()

    def train_Q_7(self, state, action, next_state, reward, not_done):
        """
        train the Q function of the learned policy: \pi
        """
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target7(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * target_Q * self.gamma

        Q1, Q2 = self.critic_net7(state, action)

        # Critic loss
        critic_loss = nn.MSELoss()(Q1, target_Q) + nn.MSELoss()(Q2, target_Q)

        # Optimize Critic
        self.critic_optim7.zero_grad()
        critic_loss.backward()
        self.critic_optim7.step()
        return critic_loss.cpu().detach().numpy().item()

    def train_actor(self, state, action):
        """
        train the learned policy
        """
        # Actor loss
        action_pi = self.actor_net(state)
        Q2 = self.critic_net2.Q1(state, action_pi)
        Q3 = self.critic_net3.Q1(state, action_pi)
        Q5 = self.critic_net5.Q1(state, action_pi)
        Q7 = self.critic_net7.Q1(state, action_pi)

        lmbda1 = self.alpha / Q2.abs().mean().detach()
        lmbda2 = self.alpha / Q3.abs().mean().detach()
        lmbda3 = self.alpha / Q5.abs().mean().detach()
        lmbda4 = self.alpha / Q7.abs().mean().detach()

        bc_loss = nn.MSELoss()(action_pi, action)

        # random
        # Q_num = random.randint(0, 3)
        # if Q_num == 0:
        #     q_loss = -lmbda1 * Q2.mean()
        # elif Q_num == 1:
        #     q_loss = -lmbda2 * Q3.mean()
        # elif Q_num == 2:
        #     q_loss = -lmbda3 * Q5.mean()
        # else:
        #     q_loss = -lmbda4 * Q7.mean()
        # Q_all = torch.cat([lmbda1 * Q2, lmbda2 * Q3, lmbda3 * Q5, lmbda4 * Q7], dim=1)

        # mean
        # q_loss = (-lmbda1 * Q2.mean() - lmbda2 * Q3.mean() - lmbda3 * Q5.mean() - lmbda4 * Q7.mean()) / 4
        # Q_all = torch.cat([lmbda1*Q2, lmbda2*Q3, lmbda3*Q5, lmbda4*Q7], dim=1)

        # max
        Q_all = torch.cat([lmbda1 * Q2, lmbda2 * Q3, lmbda3 * Q5, lmbda4 * Q7], dim=1)
        q_loss = -Q_all.max(1)[0].mean()

        # min
        # Q_all = torch.cat([lmbda1 * Q2, lmbda2 * Q3, lmbda3 * Q5, lmbda4 * Q7], dim=1)
        # q_loss = -Q_all.min(1)[0].mean()

        # nothing
        # q_loss = (-lmbda2 * Q2.mean() - lmbda3 * Q3.mean() - lmbda4 * Q4.mean()) / 3
        # Q_all = torch.cat([lmbda2*Q2, lmbda3*Q3, lmbda4*Q4], dim=1)

        consistent_loss = torch.std(Q_all, dim=1).mean()
        actor_loss = q_loss + consistent_loss

        # Optimize Actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # update the frozen target models
        for param, target_param in zip(self.critic_net2.parameters(), self.critic_target2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_net3.parameters(), self.critic_target3.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_net5.parameters(), self.critic_target5.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic_net7.parameters(), self.critic_target7.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor_net.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.cpu().detach().numpy().item(), \
               bc_loss.cpu().detach().numpy().item(), \
               Q2.mean().cpu().detach().numpy().item(), \
               Q3.mean().cpu().detach().numpy().item(), \
               Q5.mean().cpu().detach().numpy().item(), \
               Q7.mean().cpu().detach().numpy().item()

    def rollout_evaluate(self):
        """
        policy evaluation function
        :return: the evaluation result
        """
        ep_rews = 0.
        state = self.env.reset()
        while True:
            state = (state - self.s_mean) / (self.s_std + 1e-5)
            state = state.squeeze()
            action = self.actor_net(state).cpu().detach().numpy()
            state, reward, done, _ = self.env.step(action)
            # self.env.render()
            # time.sleep(0.01)
            ep_rews += reward
            if done:
                break
        ep_rews = d4rl.get_normalized_score(env_name=self.env_name, score=ep_rews) * 100
        return ep_rews

    # def save_parameters(self):
    #     torch.save(self.critic_net.state_dict(), self.file_loc[2])
    #     torch.save(self.actor_net.state_dict(), self.file_loc[3])
    #
    # def load_parameters(self):
    #     self.critic_net.load_state_dict(torch.load(self.file_loc[2]))
    #     self.actor_net.load_state_dict(torch.load(self.file_loc[3]))
