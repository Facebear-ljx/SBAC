import copy
import torch
import torch.nn as nn
import wandb
import gym
import os
import d4rl
from Sample_Dataset.Sample_from_dataset import ReplayBuffer
from Sample_Dataset.Prepare_env import prepare_env
from Network.Actor_Critic_net import Actor_deterministic, Actor, Double_Critic, EBM
from tqdm import tqdm
import datetime

EPSILON = 1e-20


class Energy:
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
                 negative_samples=256,
                 negative_policy=10,
                 batch_size=256,
                 energy_steps=int(5e+5),
                 strong_contrastive=False,
                 scale_state='minmax',
                 scale_action=True,
                 lmbda_max=100.,
                 lmbda_ood=0.15,
                 lmbda_thre=0.,
                 lr_ebm=1e-7,
                 lr_actor=3e-4,
                 lr_critic=3e-4,
                 initial_alpha=6.,
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
        super(Energy, self).__init__()

        # hyper-parameters
        self.strong_contrastive = strong_contrastive
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.policy_freq = policy_freq
        self.evaluate_freq = evaluate_freq
        self.evaluate_episodes = evalute_episodes
        self.energy_steps = energy_steps
        self.noise_clip = noise_clip
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
        if 'antmaze' in env_name and sparse == False:
            scale_rewards = True
        else:
            scale_rewards = False
        self.s_mean, self.s_std = self.replay_buffer.convert_D4RL(
            self.dataset,
            scale_rewards=scale_rewards,
            scale_state=scale_state,
            scale_action=scale_action,
            norm_reward=True
        )

        # prepare the actor and critic
        self.actor_net = Actor_deterministic(num_state, num_action, num_hidden, device).float().to(device)
        self.actor_target = copy.deepcopy(self.actor_net)
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=lr_actor)

        self.critic_net = Double_Critic(num_state, num_action, num_hidden, device).float().to(device)
        self.critic_target = copy.deepcopy(self.critic_net)
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr=lr_critic)

        self.ebm = EBM(num_state, num_action, 256, device, self.batch_size, negative_samples,
                       negative_policy).float().to(device)
        self.ebm_optim = torch.optim.Adam(self.ebm.parameters(), lr=lr_ebm)

        # Q and Critic file location
        self.current_time = datetime.datetime.now()
        logdir_name = f"./Model/{self.env_name}/{self.current_time}+{self.seed}"
        os.makedirs(logdir_name)

    # def warm_up(self, warm_time_step=1e+5):
    #     for total_it in tqdm(range(1, int(warm_time_step) + 1)):
    #         # sample data
    #         state, action, _, _, _, _ = self.replay_buffer.sample(self.batch_size)
    #
    #         # update Distance
    #         self.train_Distance(state, action)

    def learn(self, total_time_step=1e+5):
        """
        TD3_BC's learning framework
        :param total_time_step: the total iteration times for training TD3_BC
        :return: None
        """
        # self.warm_up(1e+6)

        for total_it in tqdm(range(1, int(total_time_step) + 1)):

            # sample data
            state, action, next_state, _, reward, not_done = self.replay_buffer.sample(self.batch_size)

            # update distance
            # self.train_Distance(state, action)

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

    # def train_Q_pi(self, state, action, next_state, reward, not_done):
    #     """
    #     train the Q function of the learned policy: \pi
    #     """
    #     with torch.no_grad():
    #         # Select action according to policy and add clipped noise
    #         next_action = self.actor_target.get_action(next_state)
    #         # Compute the target Q value
    #         target_O1, target_O2 = self.critic_target(next_state, next_action)
    #         distance_target = self.ebm.energy(next_state, next_action)
    #         target_Q1, target_Q2 = target_O1 - self.alpha * distance_target, target_O2 - self.alpha * distance_target
    #         target_Q = torch.min(target_Q1, target_Q2)
    #         # target_Q = torch.min(target_O1, target_O2)
    #         target_Q = reward + not_done * self.gamma * target_Q
    #
    #     O1, O2 = self.critic_net(state, action)
    #     distance = self.ebm.energy(state, action)
    #     Q1, Q2 = O1 - self.alpha * distance, O2 - self.alpha * distance
    #
    #     # Critic loss
    #     action_pi = self.actor_net.get_action(state)
    #     O1_pi, O2_pi = self.critic_net(state, action_pi)
    #
    #     grad = torch.autograd.grad(O1_pi.sum(), action_pi, retain_graph=True)[0] ** 2 \
    #            + torch.autograd.grad(O2_pi.sum(), action_pi, retain_graph=True)[0] ** 2
    #     grad_loss = torch.mean(grad)
    #
    #     # TODO \gradient w.r.t a\sim \pi, rather than a \sim Uniform(A)
    #     critic_loss = nn.MSELoss()(Q1, target_Q) + nn.MSELoss()(Q2, target_Q)
    #     final_loss = critic_loss + grad_loss * self.lmbda_min
    #
    #     # Optimize Critic
    #     self.critic_optim.zero_grad()
    #     final_loss.backward()
    #     # torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 5.0)
    #     self.critic_optim.step()
    #     return critic_loss.cpu().detach().numpy().item(), grad_loss.cpu().detach().numpy().item()

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

    # def train_Distance(self, state, action):
    #     predict, label = self.ebm.linear_distance(state, action)
    #     ebm_loss = nn.MSELoss()(predict, label)
    #
    #     self.ebm_optim.zero_grad()
    #     ebm_loss.backward()
    #     self.ebm_optim.step()
    #     return ebm_loss.cpu().detach().numpy().item()

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
                action = self.actor_net(state).cpu().detach().numpy()
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
