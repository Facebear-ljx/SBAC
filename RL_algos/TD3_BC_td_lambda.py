import copy
import torch
import torch.nn as nn
import wandb
import gym
import os
import d4rl
from Sample_Dataset.Sample_from_dataset import ReplayBuffer
from Sample_Dataset.Prepare_env import prepare_env
from Network.Actor_Critic_net import Actor_deterministic, Double_Critic


class TD3_BC_td_lambda:
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
        super(TD3_BC_td_lambda, self).__init__()
        # prepare the environment
        self.env_name = env_name
        self.env = gym.make(env_name)
        num_state = self.env.observation_space.shape[0]
        num_action = self.env.action_space.shape[0]

        # get dataset 1e+6 samples
        self.dataset = self.env.get_dataset()
        self.replay_buffer = ReplayBuffer(state_dim=num_state, action_dim=num_action, device=device)
        self.dataset = self.replay_buffer.split_dataset(self.env, self.dataset, ratio=ratio)
        self.s_mean, self.s_std = self.replay_buffer.convert_D4RL_td_lambda(self.dataset, scale_rewards=False,
                                                                            scale_state=True, n=lmbda)

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
        self.evaluate_freq = 3000
        self.noise_clip = noise_clip
        self.alpha = alpha
        self.batch_size = 256
        self.device = device
        self.max_action = 1.

        self.total_it = 0

        # Q and Critic file location
        self.file_loc = prepare_env(env_name)

    def learn(self, total_time_step=1e+5):
        """
        TD3_BC's learning framework
        :param total_time_step: the total iteration times for training TD3_BC
        :return: None
        """
        while self.total_it <= total_time_step:
            self.total_it += 1

            # sample data
            state, action, next_state, next_action, reward, not_done = self.replay_buffer.sample_lambda(self.batch_size)

            # update Critic
            critic_loss_pi = self.train_Q_pi(state, action, next_state, reward, not_done)

            # delayed policy update
            if self.total_it % self.policy_freq == 0:
                actor_loss, bc_loss, Q_pi_mean = self.train_actor(state, action)

                if self.total_it % self.evaluate_freq == 0:
                    evaluate_reward = self.rollout_evaluate()
                    wandb.log({"actor_loss": actor_loss,
                               "bc_loss": bc_loss,
                               "Q_pi_loss": critic_loss_pi,
                               "Q_pi_mean": Q_pi_mean,
                               "evaluate_rewards": evaluate_reward,
                               "it_steps": self.total_it
                               })

            if self.total_it % 100000 == 0:
                self.save_parameters()

        self.total_it = 0

    def train_Q_pi(self, state, action, next_state, reward, not_done):
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

    def train_actor(self, state, action):
        """
        train the learned policy
        """
        # Actor loss
        action_pi = self.actor_net(state)
        Q_pi = self.critic_net.Q1(state, action_pi)

        lmbda = self.alpha / Q_pi.abs().mean().detach()

        bc_loss = nn.MSELoss()(action_pi, action)

        actor_loss = -lmbda * Q_pi.mean() + bc_loss

        # Optimize Actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # update the frozen target models
        for param, target_param in zip(self.critic_net.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor_net.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.cpu().detach().numpy().item(), \
               bc_loss.cpu().detach().numpy().item(), \
               Q_pi.mean().cpu().detach().numpy().item()

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
