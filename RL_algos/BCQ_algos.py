import copy
import torch
import torch.nn as nn
import wandb
import gym
import os
import d4rl
from Sample_Dataset.Sample_from_dataset import ReplayBuffer
from Sample_Dataset.Prepare_env import prepare_env
from Network.Actor_Critic_net import Actor_deterministic, Double_Critic, BC_VAE


class BCQ:
    def __init__(self,
                 env_name,
                 num_hidden=256,
                 gamma=0.99,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2,
                 phi=0.05,
                 ratio=1,
                 lmbda=0.75,
                 device='cpu'):
        """
        Facebear's implementation of BCQ (Off-Policy Deep Reinforcement Learning without Exploration)
        Paper: http://proceedings.mlr.press/v97/fujimoto19a/fujimoto19a.pdf

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
        super(BCQ, self).__init__()
        # prepare the environment
        self.env_name = env_name
        self.env = gym.make(env_name)
        num_state = self.env.observation_space.shape[0]
        num_action = self.env.action_space.shape[0]
        num_latent = num_action * 2

        # get dataset 1e+6 samples
        self.dataset = self.env.get_dataset()
        self.replay_buffer = ReplayBuffer(state_dim=num_state, action_dim=num_action, device=device)
        self.dataset = self.replay_buffer.split_dataset(self.env, self.dataset, ratio=ratio)
        self.s_mean, self.s_std = self.replay_buffer.convert_D4RL(self.dataset, scale_rewards=False, scale_state=True)

        # prepare the actor and critic
        self.actor_net = Actor_deterministic(num_state, num_action, num_hidden, device).float().to(device)
        self.actor_target = copy.deepcopy(self.actor_net)
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=1e-3)

        self.vae = BC_VAE(num_state, num_action, num_hidden, num_latent, device).float().to(device)
        self.vae_optim = torch.optim.Adam(self.vae.parameters())

        self.critic_net = Double_Critic(num_state, num_action, num_hidden*3, device).float().to(device)
        self.critic_target = copy.deepcopy(self.critic_net)
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr=1e-3)

        # hyper-parameters
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.policy_freq = policy_freq
        self.evaluate_freq = 3000
        self.noise_clip = noise_clip
        self.phi = phi
        self.lmbda = lmbda
        self.batch_size = 256
        self.device = device
        self.max_action = 1.

        self.total_it = 0

        # Q and Critic file location
        self.file_loc = prepare_env(env_name)

    def learn(self, total_time_step=1e+5):
        """
        BCQ's learning framework
        :param total_time_step: the total iteration times for training TD3_BC
        :return: None
        """
        while self.total_it <= total_time_step:
            self.total_it += 1

            # sample data
            state, action, next_state, next_action, reward, not_done = self.replay_buffer.sample(self.batch_size)

            # update Conditional-VAE
            vae_loss = self.train_vae(state, action)

            # update Critic
            critic_loss_pi = self.train_Q_pi(state, action, next_state, reward, not_done)

            actor_loss, Q_pi_mean = self.train_actor(state)

            if self.total_it % self.evaluate_freq == 0:
                evaluate_reward = self.rollout_evaluate()
                wandb.log({"actor_loss": actor_loss,
                           "vae_loss": vae_loss,
                           "Q_pi_loss": critic_loss_pi,
                           "Q_pi_mean": Q_pi_mean,
                           "evaluate_rewards": evaluate_reward,
                           "it_steps": self.total_it
                           })

            # if self.total_it % 100000 == 0:
            #     self.save_parameters()

        self.total_it = 0

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
        most of the code refer to BCQ' s implementation
        https://github.com/sfujim/BCQ/blob/master/continuous_BCQ/BCQ.py
        """
        with torch.no_grad():
            # Duplicate next state 10 times
            next_state = torch.repeat_interleave(next_state, 10, 0)

            # Compute value of perturbed actions sampled from the VAE
            next_action = self.actor_target(next_state) * self.phi + self.vae.decode(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Soft Clipped Double Q-learning
            target_Q = self.lmbda * torch.min(target_Q1, target_Q2) + (1. - self.lmbda) * torch.max(target_Q1,
                                                                                                    target_Q2)

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

    def train_actor(self, state):
        """
        train the learned policy
        """
        # Perturbation Action
        sampled_actions = self.vae.decode(state)
        perturbation = self.actor_net(state)
        perturbed_actions = perturbation * self.phi + sampled_actions

        # Actor loss
        Q_pi = self.critic_net.Q1(state, perturbed_actions)
        actor_loss = -Q_pi.mean()

        # Optimize Actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # update the frozen target models
        for param, target_param in zip(self.critic_net.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor_net.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.cpu().detach().numpy().item(), Q_pi.mean().cpu().detach().numpy().item()

    def rollout_evaluate(self):
        """
        policy evaluation function
        :return: the evaluation result
        """
        ep_rews = 0.
        state = self.env.reset()
        while True:
            state = (state - self.s_mean) / (self.s_std + 1e-5)
            perturbation = self.actor_net(state).cpu().detach().numpy()
            action = self.vae.decode(state).cpu().detach().numpy()
            perturbed_action = perturbation * self.phi + action
            state, reward, done, _ = self.env.step(perturbed_action)
            ep_rews += reward
            if done:
                break
        ep_rews = d4rl.get_normalized_score(env_name=self.env_name, score=ep_rews) * 100
        return ep_rews

    def save_parameters(self):
        torch.save(self.critic_net.state_dict(), self.file_loc[2])
        torch.save(self.actor_net.state_dict(), self.file_loc[3])

    def load_parameters(self):
        self.critic_net.load_state_dict(torch.load(self.file_loc[2]))
        self.actor_net.load_state_dict(torch.load(self.file_loc[3]))
