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
from torch.optim.lr_scheduler import MultiStepLR
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
        self.dataset = self.replay_buffer.split_dataset(self.env, self.dataset, ratio=ratio, env_name=env_name)
        if 'antmaze' in env_name:
            scale_rewards = True
        else:
            scale_rewards = False

        self.s_mean, self.s_std = self.replay_buffer.convert_D4RL_iql(
            self.dataset,
            scale_rewards=scale_rewards,
            scale_state=scale_state,
            scale_action=scale_action,
            norm_reward=False
        )

        # prepare the actor and critic
        self.actor_net = Actor_multinormal(num_state, num_action, num_hidden, device).float().to(device)
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=lr_actor)

        self.critic_net = Double_Critic(num_state, num_action, num_hidden, device).float().to(device)
        self.critic_target = copy.deepcopy(self.critic_net)
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr=lr_critic)

        self.bc = Actor_multinormal(num_state, num_action, num_hidden, device).float().to(device)
        self.bc_optim = torch.optim.Adam(self.bc.parameters(), lr=lr_bc)
        self.bc_scheduler = MultiStepLR(self.bc_optim, milestones=[800_000, 900_000], gamma=0.1)

        self.log_alpha = torch.tensor(np.log(1.0), requires_grad=True)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -num_action

        self.bc_log_alpha = torch.tensor(np.log(1.0), requires_grad=True)
        self.bc_alpha_optim = torch.optim.Adam([self.bc_log_alpha], lr=3e-4)

        # Q and Critic file location
        self.current_time = datetime.datetime.now()
        logdir_name = f"./Model/{self.env_name}/{self.current_time}+{self.seed}"
        # os.makedirs(logdir_name)

    def warm_up(self, warm_time_step=1e+5, evaluate=False):
        bc_model_save_path = f"./Model/{self.env_name}/fisherbrc/bc/multivariate_normal_bc_alpha.pth"
        bc_save_dir = f"./Model/{self.env_name}/fisherbrc/bc/"
        if not os.path.exists(bc_model_save_path):
            print(f"pretrain bc model not found, start pretrain and save the model to {bc_model_save_path}")
            for total_it in tqdm(range(1, int(warm_time_step) + 1)):
                state, action, _, _, _ = self.replay_buffer.sample_w_o_next_acion(self.batch_size)

                bc_loss, bc_alpha_loss, entropy = self.train_bc(state, action)

                if evaluate:
                    if total_it % 10000 == 0:
                        evaluate_reward = self.rollout_evaluate_bc()
                        wandb.log({
                                   "bc_reward": evaluate_reward,
                                   "bc_loss": bc_loss.cpu().detach().numpy().item(),
                                   "bc_alpha_loss": bc_alpha_loss.cpu().detach().numpy().item(),
                                   "entropy": entropy.mean().cpu().detach().numpy().item(),
                                   "bc_alpha": self.bc_alpha().cpu().detach().numpy().item(),
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
        self.warm_up(self.warmup_steps, evaluate=True)

        for total_it in tqdm(range(1, int(total_time_step) + 1)):

            # sample data
            # sample_start_time = datetime.datetime.now()
            state, action, next_state, reward, not_done = self.replay_buffer.sample_w_o_next_acion(self.batch_size)
            reward = reward + self.reward_bonus
            # sample_end_time = datetime.datetime.now()
            # sample_time_duration = (sample_end_time - sample_start_time).microseconds

            # update Critic
            # critic_start_time = datetime.datetime.now()
            critic_loss, \
            critic_loss_in, \
            grad_loss = self.train_Q(state, action, next_state, reward, not_done)
            # critic_end_time = datetime.datetime.now()
            # critic_time = (critic_end_time - critic_start_time).microseconds

            # update actor
            # actor_start_time = datetime.datetime.now()
            actor_loss, alpha_loss, Q_pi = self.train_actor_with_auto_alpha(state)
            # actor_end_time = datetime.datetime.now()
            # actor_time = (actor_end_time - actor_start_time).microseconds

            # wandb.log({"sample_time": sample_time_duration,
                       # "critic_time": critic_time,
                       # "actor_time": actor_time})

            if total_it % self.evaluate_freq == 0:
                evaluate_reward = self.rollout_evaluate()
                wandb.log({"actor_loss": actor_loss.cpu().detach().numpy().item(),
                           "critic_loss": critic_loss.cpu().detach().numpy().item(),
                           "critic_loss_in": critic_loss_in.cpu().detach().numpy().item(),
                           "grad_norm_mean": grad_loss.cpu().detach().numpy().item()/2,
                           "Q_pi_mean": Q_pi.mean().cpu().detach().numpy().item(),
                           "evaluate_rewards": evaluate_reward,
                           "alpha": self.alpha().cpu().detach().numpy().item(),
                           "alpha_loss": alpha_loss.cpu().detach().numpy().item(),
                           # "log_pi": log_pi.mean().cpu().detach().numpy().item(),
                           "it_steps": total_it,
                           })

                # if total_it % (self.evaluate_freq * 2) == 0:
                #     self.save_parameters(evaluate_reward)
            self.total_it += 1

    def train_Q(self, state, action, next_state, reward, not_done):
        """
        train the Q function: Q(s,a) = O(s,a) + log \mu(s,a)
        """
        # in-distribution loss
        with torch.no_grad():
            next_action = self.actor_net.get_action(next_state)
            # start1 = datetime.datetime.now()  # 1
            target_Q = self.get_Q_target(next_state, next_action)  # 2
            # start2 = datetime.datetime.now()
            target_Q = reward + not_done * self.gamma * target_Q

        # start3 = datetime.datetime.now()
        Q1, Q2 = self.get_Q(state, action)  # 3
        # start4 = datetime.datetime.now()
        critic_loss_in = nn.MSELoss()(Q1, target_Q) + nn.MSELoss()(Q2, target_Q)

        # gradient loss
        # dis_pi = self.actor_net.get_dist(state)
        # policy_action = dis_pi.sample().to(self.device)
        # policy_action.requires_grad_(True)
        # start5 = datetime.datetime.now()
        policy_action = self.actor_net.get_action(state).detach()
        policy_action.requires_grad_(True)
        # start6 = datetime.datetime.now()
        O1_ood, O2_ood = self.critic_net(state, policy_action)

        O1_grads = torch.square(torch.autograd.grad(O1_ood.sum() + O2_ood.sum(), policy_action,  create_graph=True)[0])
        # O2_grads = torch.autograd.grad(O2_ood.sum(), policy_action)[0] ** 2
        O1_grads_norm = torch.sum(O1_grads, dim=1)
        grad_loss = torch.mean(O1_grads_norm)

        # final loss
        critic_loss = critic_loss_in + grad_loss * self.lmbda

        # Optimize Critic
        # critic_start = datetime.datetime.now()
        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optim.step()
        # critic_end = datetime.datetime.now()

        # print((start2-start1).microseconds, (start4-start3).microseconds, (start6-start5).microseconds, (critic_end - start1).microseconds, (critic_end - critic_start).microseconds)
        # critic_optimize_time = (critic_end - critic_start).microseconds
        # wandb.log({
        #     "critic_optimize_time": critic_optimize_time
        # })
        # update frozen target critic network
        for param, target_param in zip(self.critic_net.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss, critic_loss_in, grad_loss

    def get_Q_target(self, state, action):
        target_O1, target_O2 = self.critic_target(state, action)
        target_O = torch.min(target_O1, target_O2)
        log_mu = self.bc.get_log_density(state, action).detach()
        # assert (log_mu.mean() >= -100)
        target_Q = target_O + log_mu
        return target_Q

    def get_Q(self, state, action):
        O1, O2 = self.critic_net(state, action)
        # dis_mu = self.bc.get_dist(state)
        # action = torch.clip(action, min=-self.max_action+1e-5, max=self.max_action-1e-5)
        # log_mu = torch.unsqueeze(dis_mu.log_prob(action).detach(), dim=1)
        log_mu = self.bc.get_log_density(state, action).detach()
        # assert (log_mu.mean() >= -100)
        Q1 = O1 + log_mu
        Q2 = O2 + log_mu
        return Q1, Q2

    def get_min_Q(self, state, action):
        O1, O2 = self.critic_net(state, action)
        min_O = torch.min(O1, O2)
        log_mu = self.bc.get_log_density(state, action)
        min_Q = min_O + log_mu
        # assert (log_mu.mean() >= -100)
        return min_Q

    def get_one_Q(self, state, action):
        O1 = self.critic_net.Q1(state, action)
        log_mu = self.bc.get_log_density(state, action)
        return O1 + log_mu

    def train_actor_with_auto_alpha(self, state):
        """
           train the learned policy
        """
        # Actor loss
        # action_pi, log_pi, _ = self.actor_net(state)
        # action_pi = dis_pi.rsample()
        # action_pi = torch.clip(action_pi, min=-self.max_action+1e-5, max=self.max_action-1e-5)
        # log_pi = torch.unsqueeze(dis_pi.log_prob(action_pi), dim=1)
        # action_pi = action_pi
        action_pi, log_pi, _ = self.actor_net(state)
        # action_pi = a_distribution.rsample()
        # log_pi = a_distribution.log_prob(action_pi)
        Q_pi = self.get_one_Q(state, action_pi)
        # assert (log_pi.mean() >= -100)
        # time_start = datetime.datetime.now()
        actor_loss = torch.mean(self.alpha().detach() * log_pi - Q_pi)

        alpha_loss = self.alpha() * torch.mean(-log_pi - self.target_entropy).detach().item()
        # time_cal_in_actor = datetime.datetime.now()
        # Optimize Actor
        self.actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), 5.0)
        self.actor_optim.step()

        # start_alpha = datetime.datetime.now()
        # Optimize alpha (the auto-adjust temperature in SAC)
        self.alpha_optim.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optim.step()

        # time_end = datetime.datetime.now()
        # time = (time_end - time_start).microseconds
        # time_calculation = (time_cal_in_actor - time_start).microseconds
        # time_alpha_update = (time_end - start_alpha).microseconds
        # wandb.log({"actor_optimize_time": time,
                   # "time_alpha_update": time_alpha_update
                   # "calculation_in_actor": time_calculation
                   # })

        return actor_loss, alpha_loss, Q_pi

    def alpha(self):
        return torch.exp(self.log_alpha)

    def bc_alpha(self):
        return torch.exp(self.bc_log_alpha)

    def train_bc(self, state, action):
        # bc_dist = self.bc.get_dist(state)
        #
        # action = torch.clip(action, min=-self.max_action+1e-5, max=self.max_action-1e-5)
        # log_mu = bc_dist.log_prob(action)
        #
        # sampled_actions = bc_dist.sample()
        # sampled_actions = torch.clip(sampled_actions, min=-self.max_action+1e-5, max=self.max_action-1e-5)

        log_mu = self.bc.get_log_density(state, action)
        sampled_actions, log_pi, _ = self.bc(state)

        entropy = - log_pi

        bc_loss = - torch.mean(log_mu + self.bc_alpha().detach() * entropy)

        bc_alpha_loss = self.bc_alpha() * torch.mean(entropy - self.target_entropy).item()

        self.bc_optim.zero_grad()
        bc_loss.backward()
        self.bc_optim.step()
        self.bc_scheduler.step()

        self.bc_alpha_optim.zero_grad()
        bc_alpha_loss.backward()
        self.bc_alpha_optim.step()

        return bc_loss, bc_alpha_loss, entropy

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
