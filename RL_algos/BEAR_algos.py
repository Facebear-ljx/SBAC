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
from Network.Actor_Critic_net import Actor, BC_VAE, Ensemble_Critic

ALPHA_MAX = 500.
ALPHA_MIN = 0.2
thresh = 0.05


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
                 num_hidden=400,
                 lr_actor=3e-4,
                 lr_critic=1e-3,
                 gamma=0.99,
                 tau=0.005,
                 num_sample_batch=10,
                 warmup_steps=10000,
                 alpha=100,
                 auto_alpha=True,
                 batch_size=256,
                 num_q=4,
                 lmbda=1,
                 lagrange_thresh=10,
                 delta_conf=0.1,
                 kernel_type='laplacian',
                 device='cpu'):
        """
        Facebear's implementation of BEAR (Stabilizing Off-Policy Q-Learning via Bootstrapping error Reduction)
        Paper:https://proceedings.neurips.cc/paper/2019/file/c2073ffa77b5357a498057413bb09d3a-Paper.pdf

        :param env_name: your gym environment name
        :param num_hidden: the number of the units of the hidden layer of your network
        :param lr_actor: learning rate of the actor network
        :param lr_critic: learning rate of the critic network
        :param gamma: discounting factor of the cumulated reward
        :param warmup_steps: the steps of the training times of behavior cloning
        :param alpha: the hyper-parameters
        :param auto_alpha: whether auto update the alpha
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

        self.critic = Ensemble_Critic(num_state, num_action, num_hidden, 4, device).float().to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr_critic)

        self.vae = BC_VAE(num_state, num_action, num_hidden, num_latent, device).float().to(device)
        self.vae_optim = optim.Adam(self.vae.parameters(), lr_actor)

        # hyper-parameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.auto_alpha = auto_alpha
        self.alpha = alpha
        self.num_sample_batch = num_sample_batch
        self.num_q = num_q
        self.kernel_type = kernel_type
        self.delta_conf = delta_conf
        self.lagrange_thresh = lagrange_thresh
        self.device = device
        self.file_loc = prepare_env(env_name)
        self.total_it = 0

        if auto_alpha is True:
            self.lagrange = torch.randn((), requires_grad=True, device=device)
            self.lagrange_optim = torch.optim.Adam([self.lagrange, ], lr=1e-3)

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

            # train Critic
            critic_loss = self.train_Q_pi(state, action, next_state, reward, not_done)

            # train Actor
            actor_loss = self.train_actor(state, self.num_sample_batch)

            # Auto-update the lagrange multiplier
            self.auto_update_lagrange(state, self.num_sample_batch)

            # log_summary
            if self.total_it % 3000 == 0:
                pi_reward = self.rollout_evaluate(pi='pi')
                miu_reward = self.rollout_evaluate(pi='miu')

                wandb.log({"actor_loss": actor_loss,
                           "vae_loss": vae_loss,
                           "critic_loss": critic_loss,
                           "pi_reward": pi_reward,
                           "miu_reward": miu_reward,
                           "it_steps": self.total_it,
                           })

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
            # Algorithm 1 step 4
            next_state = torch.repeat_interleave(next_state, 10, 0)
            next_action = self.actor_target.get_action(next_state)
            target_Q_all = self.critic_target(next_state, next_action)

            # Algorithm 1 step 5
            target_Q = 0.75 * target_Q_all.min(0)[0] + 0.25 * target_Q_all.max(0)[0]
            target_Q = target_Q.view(self.batch_size, -1).max(1)[0].view(-1, 1)
            target_Q = reward + not_done * self.gamma * target_Q

        Q_all = self.critic(state, action, with_var=False)

        # Critic loss
        critic_loss = nn.MSELoss()(Q_all[0], target_Q) \
                      + nn.MSELoss()(Q_all[1], target_Q) \
                      + nn.MSELoss()(Q_all[2], target_Q) \
                      + nn.MSELoss()(Q_all[3], target_Q)

        # Optimize Critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        return critic_loss.cpu().detach().numpy().item()

    def train_actor(self, state, num_samples):
        """
        train Actor network
        :param state:
        :param num_samples:
        :return:
        """
        sampled_actions, raw_sampled_actions = self.vae.decode_multiple(state)  # 256 x 10 x 11
        actor_actions, raw_actor_actions = self.actor.get_action_multiple(state, num_samples)  # 256 x 10 x 3

        if self.kernel_type == 'gaussian':
            mmd_loss = mmd_loss_gaussian(raw_sampled_actions, raw_actor_actions)
        else:
            mmd_loss = mmd_loss_laplacian(raw_sampled_actions, raw_actor_actions)

        action_divergence = ((sampled_actions - actor_actions) ** 2).sum(-1)
        raw_action_divergence = ((raw_sampled_actions - raw_actor_actions) ** 2).sum(-1)

        critic_state_input = state.unsqueeze(0).repeat(num_samples, 1, 1)  # 10 x 256 x 11
        critic_state_input = critic_state_input.view(num_samples * state.shape[0], state.shape[1])  # 2560 x 11

        critic_action_input = actor_actions.permute(1, 0, 2).contiguous()  # 10 x 256 x 3
        critic_action_input = critic_action_input.view(num_samples * actor_actions.size(0), actor_actions.size(2))

        q_all = self.critic(critic_state_input, critic_action_input)  # 4 x 2560 x 1
        q_all = q_all.view(self.num_q, num_samples, actor_actions.shape[0], 1)  # 4 x 10 x 256 x 1
        q_all = q_all.mean(1)  # 4 x 256 x 1
        std_q = torch.std(q_all, dim=0, keepdim=False, unbiased=False)  # 256 x 1

        q_all = q_all.min(0)[0]  # 256 x 1

        # Actor loss
        if self.total_it >= self.warmup_steps:
            if self.auto_alpha:
                actor_loss = (-q_all + 0.4 * (np.sqrt(
                    (1 - self.delta_conf) / self.delta_conf)) * std_q + self.lagrange.exp().detach() * mmd_loss).mean()
            else:
                actor_loss = (-q_all + 0.4 * (
                    np.sqrt((1 - self.delta_conf) / self.delta_conf)) * std_q + self.alpha * mmd_loss).mean()
        else:
            if self.auto_alpha:
                actor_loss = (self.lagrange.exp().detach() * mmd_loss).mean()
            else:
                actor_loss = self.alpha * mmd_loss.mean()

        std_loss = 0.4 * (np.sqrt((1 - self.delta_conf) / self.delta_conf)) * std_q.detach()

        # Actor optim
        self.actor_optim.zero_grad()
        if self.auto_alpha:
            actor_loss.backward(retain_graph=True)
        else:
            actor_loss.backward()
        self.actor_optim.step()

        # Update Target Networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.cpu().detach().numpy().item()

    def auto_update_lagrange(self, state, num_samples):
        sampled_actions, raw_sampled_actions = self.vae.decode_multiple(state)  # 256 x 10 x 11
        actor_actions, raw_actor_actions = self.actor.get_action_multiple(state, num_samples)  # 256 x 10 x 3

        if self.kernel_type == 'gaussian':
            mmd_loss = mmd_loss_gaussian(raw_sampled_actions, raw_actor_actions)
        else:
            mmd_loss = mmd_loss_laplacian(raw_sampled_actions, raw_actor_actions)

        # Auto-update lagrange multiplier
        if self.auto_alpha:
            lagrange_loss = (self.lagrange.exp()*(mmd_loss-thresh)).mean()
            self.lagrange_optim.zero_grad()
            lagrange_loss.backward()
            self.lagrange_optim.step()
            self.lagrange.data.clamp_(min=-5.0, max=self.lagrange_thresh)

    def rollout_evaluate(self, pi='pi'):
        """
        policy evaluation function
        :param pi: you can select pi='pi' or 'miu'. If you select the pi='pi', then the policy evaluated is the learned policy
        :return: the evaluation result
        """
        ep_rews = 0.
        self.vae.eval()
        for _ in range(1):
            ep_rews = 0.
            state = self.env.reset()
            ep_t = 0
            while True:
                ep_t += 1
                if pi == 'pi':
                    state = (state - self.s_mean) / (self.s_std + 1e-5)
                    state = state.squeeze()
                    action = self.actor.get_action(state).cpu().detach().numpy()
                else:
                    state = (state - self.s_mean) / (self.s_std + 1e-5)
                    state = state.squeeze()
                    action = self.vae.decode(np.expand_dims(state, 0)).cpu().detach().numpy()
                    action = action.squeeze()
                state, reward, done, _ = self.env.step(action)
                ep_rews += reward
                if done:
                    break
        ep_rews = d4rl.get_normalized_score(env_name=self.env_name, score=ep_rews) * 100
        return ep_rews

    # def save_parameters(self):
    #     torch.save(self.critic.state_dict(), self.file_loc[0])
    #     torch.save(self.q_pi_net.state_dict(), self.file_loc[2])
    #     torch.save(self.actor_net.state_dict(), self.file_loc[3])

    # def load_parameters(self):
    #     self.bc_standard_net.load_state_dict(torch.load(self.file_loc[1]))
