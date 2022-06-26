import copy
import torch
import torch.nn as nn
import wandb
import gym
import os
import d4rl
from Sample_Dataset.Sample_from_dataset import ReplayBuffer
import numpy as np
from Network.Actor_Critic_net import Actor_multinormal, Double_Critic
from tqdm import tqdm
import datetime

EPSILON = 1e-20


class FisherBRC:
    def __init__(self,
                 env_name,
                 num_hidden=256,
                 gamma=0.995,
                 tau=0.005,
                 policy_freq=2,
                 ratio=1,
                 seed=0,
                 batch_size=256,
                 warmup_steps=int(5e+5),
                 scale_state='minmax',
                 scale_action=True,
                 lmbda=0.15,
                 reward_bonus=5.0,
                 lr_bc=3e-4,
                 lr_actor=3e-4,
                 lr_critic=3e-4,
                 evaluate_freq=5000,
                 evalute_episodes=10,
                 device='cpu'):
        """
        Facebear's implementation of Fisher_BRC
        Paper:

        :param env_name: your gym environment name
        :param num_hidden: the number of the units of the hidden layer of your network
        :param gamma: discounting factor of the cumulated reward
        :param tau: soft update
        :param policy_freq: delayed policy update frequency
        :param alpha: the hyper-parameter in equation
        :param ratio:
        :param device:
        """
        super(FisherBRC, self).__init__()

        # hyper-parameters
        self.gamma = gamma
        self.tau = tau
        self.policy_freq = policy_freq
        self.evaluate_freq = evaluate_freq
        self.evaluate_episodes = evalute_episodes
        self.warmup_steps = warmup_steps
        self.reward_bonus = reward_bonus
        self.batch_size = batch_size
        self.device = device
        self.max_action = 1.
        self.lmbda = lmbda
        self.scale_state = scale_state
        self.scale_action = scale_action
        self.total_it = 0

        # prepare the environment
        self.env_name = env_name
        self.env = gym.make(env_name)
        num_state = self.env.observation_space.shape[0]
        num_action = self.env.action_space.shape[0]

        # set seed
        self.seed = seed
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        torch.manual_seed(seed)

        # get dataset 1e+6 samples
        self.dataset = self.env.get_dataset()
        self.replay_buffer = ReplayBuffer(state_dim=num_state, action_dim=num_action, device=device)
        self.dataset = self.replay_buffer.split_dataset(self.env, self.dataset, ratio=ratio, toycase=toycase,
                                                        env_name=env_name)
        if 'antmaze' in env_name:
            scale_rewards = True
        else:
            scale_rewards = False

        self.s_mean, self.s_std = self.replay_buffer.convert_D4RL_iql(
            self.dataset,
            scale_rewards=scale_rewards,
            scale_state=scale_state,
            scale_action=scale_action,
            norm_reward=True
        )

        # prepare the actor and critic
        self.actor_net = Actor_multinormal(num_state, num_action, num_hidden, device).float().to(device)
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=lr_actor)

        self.critic_net = Double_Critic(num_state, num_action, num_hidden, device).float().to(device)
        self.critic_target = copy.deepcopy(self.critic_net)
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr=lr_critic)

        self.bc = Actor_multinormal(num_state, num_action, num_hidden, device).float().to(device)
        self.bc_optim = torch.optim.Adam(self.bc.parameters(), lr=lr_bc)

        self.log_alpha = torch.tensor(np.log(1.0), requires_grad=True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -num_action

        # Q and Critic file location
        self.current_time = datetime.datetime.now()
        logdir_name = f"./Model/{self.env_name}/{self.current_time}+{self.seed}"
        os.makedirs(logdir_name)

    def warm_up(self, warm_time_step=1e+5):
        bc_model_save_path = f"./Model/{self.env_name}/fisherbrc/bc/multivariate_normal_bc.pth"
        bc_save_dir = f"./Model/{self.env_name}/fisherbrc/bc/"
        if not os.path.exists(bc_model_save_path):
            print(f"pretrain bc model not found, start pretrain and save the model to {bc_model_save_path}")
            for total_it in tqdm(range(1, int(warm_time_step) + 1)):
                state, action, _, _, _ = self.replay_buffer.sample_w_o_next_acion(self.batch_size)

                bc_loss = self.train_bc(state, action)

                if total_it % self.evaluate_freq == 0:
                    evaluate_reward = self.rollout_evaluate_bc()
                    wandb.log({"bc_reward": evaluate_reward,
                               "bc_loss": bc_loss.cpu().detach().numpy().item(),
                               "warmup_steps": total_it})

            if not os.path.exists(bc_save_dir):
                os.makedirs(bc_save_dir)

            torch.save(self.bc.state_dict(), bc_model_save_path)
        else:
            print(f"load bc model from {bc_model_save_path}")
            self.bc.load_state_dict(torch.load(bc_model_save_path))

    def learn(self, total_time_step=1e+5):
        """
        Learning framework of Fisher_BRC
        :param total_time_step: the total iteration times for training Fisher_BRC
        :return: None
        """
        self.warm_up(self.warmup_steps)

        for total_it in tqdm(range(1, int(total_time_step) + 1)):

            # sample data
            state, action, next_state, reward, not_done = self.replay_buffer.sample_w_o_next_acion(self.batch_size)
            reward = reward + self.reward_bonus

            # update Critic
            critic_loss, critic_loss_in, grad_loss = self.train_Q(state, action, next_state, reward, not_done)

            # delayed policy update
            if total_it % self.policy_freq == 0:
                actor_loss, grad_loss, Q_pi = self.train_actor_with_auto_alpha(state, action)
                if total_it % self.evaluate_freq == 0:
                    evaluate_reward = self.rollout_evaluate()
                    wandb.log({"actor_loss": actor_loss.cpu().detach().numpy().item(),
                               "critic_loss": critic_loss.cpu().detach().numpy().item(),
                               "critic_loss_in": critic_loss_in.cpu().detach().numpy().item(),
                               "grad_loss": grad_loss.cpu().detach().numpy().item(),
                               "Q_pi_mean": Q_pi.mean().cpu().detach().numpy().item(),
                               "evaluate_rewards": evaluate_reward,
                               "it_steps": total_it,
                               })

                    if total_it % (self.evaluate_freq * 2) == 0:
                        self.save_parameters(evaluate_reward)
            self.total_it += 1

    def train_Q(self, state, action, next_state, reward, not_done):
        """
        train the Q function: Q(s,a) = O(s,a) + log \mu(s,a)
        """
        # in-distribution loss
        with torch.no_grad():
            next_action = self.actor_net.get_action(next_state)
            target_Q = self.get_Q_target(next_state, next_action)
            target_Q = reward + not_done * self.gamma * target_Q

        Q1, Q2 = self.get_Q(state, action)
        critic_loss_in = nn.MSELoss()(Q1, target_Q) + nn.MSELoss()(Q2, target_Q)

        # gradient loss
        policy_action = self.actor_net.get_action(state)
        Q1_ood, Q2_ood = self.get_Q(state, policy_action)

        Q1_grads = torch.autograd.grad(Q1_ood.sum(), policy_action)[0] ** 2
        Q2_grads = torch.autograd.grad(Q2_ood.sum(), policy_action)[0] ** 2
        grad_loss = torch.mean(Q1_grads + Q2_grads)

        # final loss
        critic_loss = critic_loss_in + grad_loss * self.lmbda

        # Optimize Critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # update frozen target critic network
        for param, target_param in zip(self.critic_net.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss, critic_loss_in, grad_loss

    def get_Q_target(self, state, action):
        target_O1, target_O2 = self.critic_target(state, action)
        target_O = torch.min(target_O1, target_O2)
        log_mu = self.bc.get_log_density(state, action)
        target_Q = target_O + log_mu
        return target_Q

    def get_Q(self, state, action):
        O1, O2 = self.critic_net(state, action)
        log_mu = self.bc.get_log_density(state, action)
        Q1 = O1 + log_mu
        Q2 = O2 + log_mu
        return Q1, Q2

    def train_actor_with_auto_alpha(self, state, action):
        """
           train the learned policy
        """
        # Actor loss
        action_pi, log_pi, _ = self.actor_net(state)
        Q1, Q2 = self.get_Q(state, action_pi)
        Q_pi = torch.min(Q1, Q2)
        actor_loss = torch.mean(self.alpha() * log_pi - Q_pi)

        alpha_loss = torch.mean(self.alpha() * (-log_pi - self.target_entropy))

        # Optimize Actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Optimize alpha (the auto-adjust temperature in SAC)
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.actor_optim.step()

        return actor_loss, alpha_loss, Q_pi

    def alpha(self):
        return torch.exp(self.log_alpha)

    def train_bc(self, state, action):
        log_pi = self.bc.get_log_density(state, action)

        bc_loss = - torch.mean(log_pi)

        self.bc_optim.zero_grad()
        bc_loss.backward()
        self.bc_optim.step()
        return bc_loss

    def rollout_evaluate(self):
        """
        policy evaluation function
        :return: the evaluation result
        """
        state = self.env.reset()
        ep_rews = 0.
        for i in range(self.evaluate_episodes):
            while True:
                if self.scale_state == 'standard':
                    state = (state - self.s_mean) / (self.s_std + 1e-5)
                elif self.scale_state == 'minmax':
                    state = (state - self.s_mean) / (self.s_std - self.s_mean)
                state = state.squeeze()
                action = self.actor_net.get_action(state).cpu().detach().numpy()
                state, reward, done, _ = self.env.step(action)
                ep_rews += reward
                if done:
                    state = self.env.reset()
                    break
        ep_rews = d4rl.get_normalized_score(env_name=self.env_name, score=ep_rews) * 100 / self.evaluate_episodes
        print('reward:', ep_rews)
        return ep_rews

    def rollout_evaluate_bc(self):
        """
                policy evaluation function
                :return: the evaluation result
                """
        state = self.env.reset()
        ep_rews = 0.
        for i in range(self.evaluate_episodes):
            while True:
                if self.scale_state == 'standard':
                    state = (state - self.s_mean) / (self.s_std + 1e-5)
                state = state.squeeze()
                action = self.bc.get_action(state).cpu().detach().numpy()
                state, reward, done, _ = self.env.step(action)
                ep_rews += reward
                if done:
                    state = self.env.reset()
                    break
        ep_rews = d4rl.get_normalized_score(env_name=self.env_name, score=ep_rews) * 100 / self.evaluate_episodes
        print('reward:', ep_rews)
        return ep_rews

    def save_parameters(self, reward):
        logdir_name = f"./Model/{self.env_name}/{self.current_time}+{self.seed}/{self.total_it}+{reward}"
        os.makedirs(logdir_name)

        # q_logdir_name = f"./Model/{self.env_name}/{self.current_time}+{self.seed}/{self.total_it}+{reward}/q.pth"
        a_logdir_name = f"./Model/{self.env_name}/{self.current_time}+{self.seed}/{self.total_it}+{reward}/actor.pth"
        # target_q_logdir_name = f"./Model/{self.env_name}/{self.current_time}+{self.seed}/{self.total_it}+{reward}/q_.pth"
        # target_a_logdir_name = f"./Model/{self.env_name}/{self.current_time}+{self.seed}/{self.total_it}+{reward}/actor_.pth"
        distance_logdir_name = f"./Model/{self.env_name}/{self.current_time}+{self.seed}/{self.total_it}+{reward}/distance.pth"
        # torch.save(self.critic_net.state_dict(), q_logdir_name)
        # torch.save(self.critic_target.state_dict(), target_q_logdir_name)
        torch.save(self.actor_net.state_dict(), a_logdir_name)
        # torch.save(self.actor_target.state_dict(), target_a_logdir_name)
        torch.save(self.ebm.state_dict(), distance_logdir_name)

    # def load_parameters(self):
    #     self.critic_net.load_state_dict(torch.load(self.file_loc[2]))
    #     self.actor_net.load_state_dict(torch.load(self.file_loc[3]))
