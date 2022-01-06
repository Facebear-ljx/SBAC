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
from Network.Actor_Critic_net import Actor, BC_VAE, Ensemble_Critic, Alpha, W

ALPHA_MAX = 500.
ALPHA_MIN = 0.2
EPS = 1e-8


# copy from kumar's implementation:
# https://github.com/aviralkumar2907/BEAR/blob/f2e31c1b5f81c4fb0e692a34949c7d8b48582d8f/algos.py#L53
def mmd_loss_laplacian(samples1, samples2, sigma=10.):
    """MMD constraint with Laplacian kernel for support matching"""
    # sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
    diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
    diff_x_x = torch.mean((-(diff_x_x.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

    diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
    diff_x_y = torch.mean((-(diff_x_y.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

    diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
    diff_y_y = torch.mean((-(diff_y_y.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

    overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
    return overall_loss


# copy from kumar's implementation:
# https://github.com/aviralkumar2907/BEAR/blob/f2e31c1b5f81c4fb0e692a34949c7d8b48582d8f/algos.py#L53
def mmd_loss_gaussian(samples1, samples2, sigma=10.):
    """MMD constraint with Gaussian Kernel support matching"""
    # sigma is set to 10.0 for hopper, cheetah and 20 for walker/ant
    diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)  # B x N x N x d
    diff_x_x = torch.mean((-(diff_x_x.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

    diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
    diff_x_y = torch.mean((-(diff_x_y.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

    diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)  # B x N x N x d
    diff_y_y = torch.mean((-(diff_y_y.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))

    overall_loss = (diff_x_x + diff_y_y - 2.0 * diff_x_y + 1e-6).sqrt()
    return overall_loss


class BEAR:
    max_grad_norm = 0.5
    soft_update = 0.005

    def __init__(self, env_name,
                 num_hidden,
                 lr_actor=1e-5,
                 lr_critic=1e-3,
                 gamma=0.99,
                 warmup_steps=10000,
                 alpha=0.2,
                 auto_alpha=True,
                 epsilon=1,
                 batch_size=256,
                 lmbda=1,
                 device='cpu'):
        """
        Facebear's implementation of BEAR (Stabilizing Off-Policy Q-Learning via Bootstrapping error Reduction)
        Paper:https://proceedings.neurips.cc/paper/2019/file/c2073ffa77b5357a498057413bb09d3a-Paper.pdf

        :param env_name: your gym environment name
        :param num_hidden: the number of the units of the hidden layer of your network
        :param Use_W: whether use the importance sampling ratio to update the policy in equation(8)
        :param lr_actor: learning rate of the actor network
        :param lr_critic: learning rate of the critic network
        :param gamma: discounting factor of the cumulated reward
        :param warmup_steps: the steps of the training times of behavior cloning
        :param alpha: the hyper-parameters in equation(8)
        :param auto_alpha: whether auto update the alpha
        :param epsilon: the constraint
        :param batch_size: sample batch size
        :param device: cuda or cpu
        """
        super(BEAR, self).__init__()

        # prepare the env and dataset
        self.env_name = env_name
        self.env = gym.make(env_name)
        num_state = self.env.observation_space.shape[0]
        num_action = self.env.action_space.shape[0]
        num_latent = num_action * 2

        # get dataset 1e+6 samples
        self.dataset = self.env.get_dataset()
        self.replay_buffer = ReplayBuffer(state_dim=num_state, action_dim=num_action, device=device)
        self.s_mean, self.s_std = self.replay_buffer.convert_D4RL_td_lambda(self.dataset, scale_rewards=False,
                                                                            scale_state=True, n=lmbda)

        # Actor, Critic, Conditional VAE
        self.actor = Actor(num_state, num_action, num_hidden, device).float().to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr_actor)

        self.critic = Ensemble_Critic(num_state, num_action, num_hidden, num_q=4, device=device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr_critic)

        self.vae = BC_VAE(num_state, num_action, num_hidden, num_latent, device).float().to(device)
        self.vae_optim = optim.Adam(self.vae.parameters(), lr_actor)

        self.file_loc = prepare_env(env_name)

        if auto_alpha is True:
            self.log_lagrange2 = torch.randn((), requires_grad=True, device=device)
            self.lagrange2_optim = torch.optim.Adam([self.log_lagrange2, ], lr=1e-3)

        # hyper-parameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.auto_alpha = auto_alpha
        self.alpha = alpha
        self.epsilon = epsilon
        self.device = device
        self.total_it = 0

    def kl_loss(self, samples1, state):
        """We just do likelihood, we make sure that the policy is close to the
           data in terms of the KL."""
        state_rep = state.unsqueeze(1).repeat(1, samples1.size(1), 1).view(-1, state.size(-1))
        samples1_reshape = samples1.view(-1, samples1.size(-1))
        samples1_log_pis = self.actor.log_pis(state=state_rep, raw_action=samples1_reshape)
        samples1_log_prob = samples1_log_pis.view(state.size(0), samples1.size(1))
        return (-samples1_log_prob).mean(1)

    def entropy_loss(self, samples1, state):
        state_rep = state.unsqueeze(1).repeat(1, samples1.size(1), 1).view(-1, state.size(-1))
        samples1_reshape = samples1.view(-1, samples1.size(-1))
        samples1_log_pis = self.actor.log_pis(state=state_rep, raw_action=samples1_reshape)
        samples1_log_prob = samples1_log_pis.view(state.size(0), samples1.size(1))
        samples1_prob = samples1_log_prob.clamp(min=-5, max=4).exp()
        return samples1_prob.mean(1)

    def select_action(self, state):
        """When running the actor, we just select action based on the max of the Q-function computed over
            samples from the policy -- which biases things to support."""
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(10, 1).to(self.device)
            action = self.actor(state)
            q1 = self.critic.q1(state, action)
            ind = q1.max(0)[1]
        return action[ind].cpu().data.numpy().flatten()

    def pretrain_bc_standard(self):
        """
        pretrain behavior cloning for self.warmup_steps and save the bc model parameters at self.file_loc
        :return: None
        """
        for i in range(self.warmup_steps):
            state, action, _, _, _, _ = self.replay_buffer.sample(self.batch_size)
            bc_loss = self.train_bc_standard(s=state, a=action)
            if i % 5000 == 0:
                miu_reward = self.rollout_evaluate(pi='miu')
                wandb.log({"bc_loss": bc_loss,
                           "miu_reward": miu_reward
                           })
        torch.save(self.vae.state_dict(), self.file_loc[1])

    def learn(self, total_time_step=1e+6):
        """
        BEAR's learning framework
        :param total_time_step: the total iteration times for training BEAR
        :return: None
        """
        # train !
        while self.total_it <= total_time_step:
            self.total_it += 1

            # sample data
            state, action, next_state, next_action, reward, not_done = self.replay_buffer.sample(self.batch_size)

            # train VAE
            vae_loss = self.train_vae(state, action)

            # update Critic
            critic_loss_pi = self.train_Q_pi(state, action, next_state, reward, not_done)

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

            # log_summary
            if self.total_it % 3000 == 0:
                pi_reward = self.rollout_evaluate(pi='pi')
                miu_reward = self.rollout_evaluate(pi='miu')

                wandb.log({"actor_loss": actor_loss.item(),
                           "vae_loss": vae_loss.item(),
                           "pi_reward": pi_reward,
                           "miu_reward": miu_reward,
                           "log_miu": log_miu.mean().item(),
                           "it_steps": self.total_it,
                           "alpha": self.alpha.item()
                           })

    def train_vae(self, state, action):
        """
        train the Conditional-VAE
        """
        action_recon, mean, std = self.vae(state, action)
        recon_loss = nn.MSELoss()(action_recon, action)
        KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss

        self.vae_optim.zero_grad()
        vae_loss.backward()
        self.vae_optim.step()
        return vae_loss.cpu().detach().numpy().item()

    def train_Q_pi(self, state, action, next_state, reward, not_done):
        """
        train the Q function of the learned policy: \pi
        most of the code refer to kumar' s implementation
        https://github.com/aviralkumar2907/BEAR/blob/f2e31c1b5f81c4fb0e692a34949c7d8b48582d8f/algos.py#L53
        """
        with torch.no_grad():
            # Duplicate next state 10 times similar to BCQ
            next_state = torch.repeat_interleave(next_state, 10, 0)

            next_action = self.actor_target(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Soft Clipped Double Q-learning
            target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1, target_Q2)

            # Take max over each action sampled from the VAE
            target_Q = target_Q.reshape(self.batch_size, -1).max(1)[0].reshape(-1, 1)
            target_Q = reward + not_done * self.gamma * target_Q

        Q1, Q2 = self.critic_net(state, action)

        # Critic loss
        critic_loss = nn.MSELoss()(Q1, target_Q) + nn.MSELoss()(Q2, target_Q)

        # Optimize Critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        return critic_loss.cpu().detach().numpy().item()

    def update_alpha(self, state, action, log_miu):
        """
        auto update the SBAC's hyper-parameter alpha in equation(8)
        """
        alpha = self.alpha_net(state, action)
        self.alpha_optim.zero_grad()
        alpha_loss = torch.mean(alpha * (log_miu + self.epsilon))
        alpha_loss.backward()
        self.alpha_optim.step()

    def get_alpha(self, state, action):
        """
        auto get the appropriate parameter alpha in equation(8)
        """
        return self.alpha_net(state, action).detach()


    def rollout_evaluate(self, pi='pi'):
        """
        policy evaluation function
        :param pi: you can select pi='pi' or 'miu'. If you select the pi='pi', then the policy evaluated is the learned policy
        :return: the evaluation result
        """
        ep_rews = 0.
        self.bc_standard_net.eval()
        for _ in range(1):
            ep_rews = 0.
            state = self.env.reset()
            ep_t = 0
            while True:
                ep_t += 1
                if pi == 'pi':
                    state = (state - self.s_mean) / (self.s_std + 1e-5)
                    state = state.squeeze()
                    action = self.actor_net.get_action(state).cpu().detach().numpy()
                else:
                    state = (state - self.s_mean) / (self.s_std + 1e-5)
                    state = state.squeeze()
                    action = self.bc_standard_net.get_action(np.expand_dims(state, 0)).cpu().detach().numpy()
                    action = action.squeeze()
                state, reward, done, _ = self.env.step(action)
                ep_rews += reward
                if done:
                    break
        ep_rews = d4rl.get_normalized_score(env_name=self.env_name, score=ep_rews) * 100
        return ep_rews

    def save_parameters(self):
        torch.save(self.q_net.state_dict(), self.file_loc[0])
        torch.save(self.q_pi_net.state_dict(), self.file_loc[2])
        torch.save(self.actor_net.state_dict(), self.file_loc[3])

    def load_parameters(self):
        self.bc_standard_net.load_state_dict(torch.load(self.file_loc[1]))

    def load_q_actor_parameters(self):
        self.bc_standard_net.load_state_dict(torch.load(self.file_loc[1]))
        self.q_net.load_state_dict(torch.load(self.file_loc[0]))
        self.actor_net.load_state_dict(torch.load(self.file_loc[3]))
