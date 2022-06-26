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
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2,
                 lmbda_in=2.5,
                 ratio=1,
                 seed=0,
                 batch_size=256,
                 warmup_steps=int(5e+5),
                 strong_contrastive=False,
                 scale_state='minmax',
                 scale_action=True,
                 lmbda_max=100.,
                 lmbda_ood=0.15,
                 lmbda_thre=0.,
                 lr_bc=3e-4,
                 lr_actor=3e-4,
                 lr_critic=3e-4,
                 toycase=False,
                 sparse=False,
                 evaluate_freq=5000,
                 evalute_episodes=10,
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
        super(FisherBRC, self).__init__()

        # hyper-parameters
        self.strong_contrastive = strong_contrastive
        self.gamma = gamma
        self.tau = tau
        self.policy_freq = policy_freq
        self.evaluate_freq = evaluate_freq
        self.evaluate_episodes = evalute_episodes
        self.warmup_steps = warmup_steps
        self.lmbda_in = lmbda_in
        self.batch_size = batch_size
        self.device = device
        self.max_action = 1.
        self.lmbda_max = lmbda_max
        self.lmbda_ood = lmbda_ood
        self.lmbda_thre = lmbda_thre
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
        self.actor_target = copy.deepcopy(self.actor_net)
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=lr_actor)

        self.critic_net = Double_Critic(num_state, num_action, num_hidden, device).float().to(device)
        self.critic_target = copy.deepcopy(self.critic_net)
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr=lr_critic)

        self.bc = Actor_multinormal(num_state, num_action, num_hidden, device).float().to(device)
        self.bc_optim = torch.optim.Adam(self.bc.parameters(), lr=lr_bc)

        self.log_alpha = torch.tensor(np.log(1.0), requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

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
        for total_it in tqdm(range(1, int(total_time_step) + 1)):

            # sample data
            state, action, next_state, reward, not_done = self.replay_buffer.sample_w_o_next_acion(self.batch_size)

            # update Critic
            critic_loss_in, critic_loss_ood = self.train_Q_dis(state, action, next_state, reward, not_done)

            # delayed policy update
            if total_it % self.policy_freq == 0:
                actor_loss, bc_loss, Q_pi_mean = self.train_actor_with_auto_alpha(state, action)
                if total_it % self.evaluate_freq == 0:
                    evaluate_reward = self.rollout_evaluate()
                    wandb.log({"actor_loss": actor_loss.cpu().detach().numpy().item(),
                               "bc_loss": bc_loss.cpu().detach().numpy().item(),
                               "critic_loss_ood": critic_loss_ood.cpu().detach().numpy().item(),
                               "critic_loss_in": critic_loss_in.cpu().detach().numpy().item(),
                               "Q_pi_mean": Q_pi_mean.cpu().detach().numpy().item(),
                               # "energy": energy,
                               # "distance_mean": energy_mean,
                               "evaluate_rewards": evaluate_reward,
                               # "dual_alpha": self.auto_alpha.detach(),
                               "it_steps": total_it,
                               # "log_barrier": log_barrier
                               })

                    if total_it % (self.evaluate_freq * 2) == 0:
                        self.save_parameters(evaluate_reward)
            self.total_it += 1

    def train_Q_dis(self, state, action, next_state, reward, not_done):
        """
        train the Q function that incorporates distance property
        """
        # in-distribution loss
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            # distance_target = self.ebm.energy(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            # target_Q = target_Q - self.lmbda_in * distance_target
            target_Q = reward + not_done * self.gamma * target_Q
        Q1, Q2 = self.critic_net(state, action)
        critic_loss_in = nn.MSELoss()(Q1, target_Q) + nn.MSELoss()(Q2, target_Q)

        # out-distribution loss
        action_ood = self.actor_net(state)
        Q1_ood, Q2_ood = self.critic_net(state, action_ood)

        # distance_ood = self.ebm.energy(state, action_ood)
        with torch.no_grad():
            target_ood = torch.min(Q1_ood, Q2_ood)
            # target_in = torch.max(Q1, Q2)
            # target_ood = target_ood - self.lmbda_ood * distance_ood
        critic_loss_ood = nn.MSELoss()(Q1_ood, target_ood) + nn.MSELoss()(Q2_ood, target_ood)
        # critic_loss_ = nn.MSELoss()(Q1, target_in) + nn.MSELoss()(Q2, target_in)
        # final loss
        critic_loss = critic_loss_in + critic_loss_ood * self.lmbda_ood

        # Optimize Critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 5.0)
        self.critic_optim.step()
        return critic_loss_in, critic_loss_ood

    def train_actor_with_auto_alpha(self, state, action):
        """
               train the learned policy
               """
        # Actor loss
        action_pi = self.actor_net(state)

        # Q1, Q2 = self.critic_net(state, action_pi)
        # Q_pi = torch.min(Q1, Q2)
        # Q_pi = Q1

        Q_pi = self.critic_net.Q1(state, action_pi)
        # distance = self.ebm.energy(state, action_pi)
        # lmbda = self.alpha / Q_pi.abs().mean().detach()

        bc_loss = nn.MSELoss()(action_pi, action)

        actor_loss = -Q_pi.mean() + bc_loss

        # Optimize Actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # update the frozen target models
        for param, target_param in zip(self.critic_net.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor_net.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss, \
               bc_loss, \
               Q_pi.mean(), \
            # distance.mean().cpu().detach().numpy().item()
        # energy_loss.mean().cpu().detach().numpy().item(), \

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
