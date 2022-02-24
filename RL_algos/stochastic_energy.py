import copy
import torch
import torch.nn as nn
import wandb
import gym
import os
import d4rl
from Sample_Dataset.Sample_from_dataset import ReplayBuffer
from Sample_Dataset.Prepare_env import prepare_env
from Network.Actor_Critic_net import Actor_deterministic, Double_Critic, EBM, Actor
from tqdm import tqdm


class Energy:
    def __init__(self,
                 env_name,
                 num_hidden=256,
                 gamma=0.99,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_freq=2,
                 alpha=2.5,
                 ratio=1,
                 seed=0,
                 negative_samples=256,
                 negative_policy=10,
                 batch_size=256,
                 energy_steps=int(5e+5),
                 strong_contrastive=False,
                 lmbda_max=100.,
                 lmbda_min=0.15,
                 lmbda_thre=0.,
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
        self.evaluate_freq = 3000
        self.energy_steps = energy_steps
        self.noise_clip = noise_clip
        self.alpha = alpha
        self.batch_size = batch_size
        self.device = device
        self.max_action = 1.
        self.lmbda_max = lmbda_max
        self.lmbda_min = lmbda_min
        self.lmbda_thre = lmbda_thre

        self.total_it = 0

        # prepare the environment
        self.env_name = env_name
        self.env = gym.make(env_name)
        num_state = self.env.observation_space.shape[0]
        num_action = self.env.action_space.shape[0]

        # set seed
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        torch.manual_seed(seed)

        # get dataset 1e+6 samples
        self.dataset = self.env.get_dataset()
        self.replay_buffer = ReplayBuffer(state_dim=num_state, action_dim=num_action, device=device)
        self.dataset = self.replay_buffer.split_dataset(self.env, self.dataset, ratio=ratio)
        self.s_mean, self.s_std = self.replay_buffer.convert_D4RL(self.dataset, scale_rewards=False, scale_state=False)

        # prepare the actor and critic
        self.actor_net = Actor(num_state, num_action, num_hidden, device).float().to(device)
        self.actor_target = copy.deepcopy(self.actor_net)
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=3e-4)

        self.critic_net = Double_Critic(num_state, num_action, num_hidden, device).float().to(device)
        self.critic_target = copy.deepcopy(self.critic_net)
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr=3e-4)

        self.ebm = EBM(num_state, num_action, num_hidden, device, self.batch_size, negative_samples,
                       negative_policy).float().to(device)
        self.ebm_optim = torch.optim.Adam(self.ebm.parameters(), lr=3e-4)
        # AAA = torch.ones(1, num_action).to(device)
        # self.lmbda = torch.tensor(AAA, dtype=torch.float, requires_grad=True)
        # self.lmbda.to(device)
        # self.lmbda_optim = torch.optim.Adam([self.lmbda], lr=1e-2)

        self.auto_alpha = torch.tensor(1., dtype=torch.float, requires_grad=True)
        self.auto_alpha.to(device)
        self.auto_alpha_optim = torch.optim.Adam([self.auto_alpha], lr=1e-2)

        # Q and Critic file location
        # self.file_loc = prepare_env(env_name)

    def learn(self, total_time_step=1e+5):
        """
        TD3_BC's learning framework
        :param total_time_step: the total iteration times for training TD3_BC
        :return: None
        """
        for total_it in tqdm(range(int(total_time_step))):

            # sample data
            state, action, next_state, next_action, reward, not_done = self.replay_buffer.sample(self.batch_size)

            # update Critic
            critic_loss_pi = self.train_Q_pi(state, action, next_state, reward, not_done)

            # update Energy
            if total_it <= self.energy_steps:
                if self.strong_contrastive:
                    ebm_loss = self.train_strong_Energy(state, action)
                else:
                    ebm_loss = self.train_Energy(state, action)
            else:
                ebm_loss = 0.

            # delayed policy update
            if total_it % self.policy_freq == 0:
                actor_loss, bc_loss, Q_pi_mean, energy, energy_mean, alpha = self.train_actor_with_auto_alpha(state, action)
                # actor_loss, bc_loss, Q_pi_mean, energy, energy_mean = self.train_actor(state, action)
                if total_it % self.evaluate_freq == 0:
                    evaluate_reward = self.rollout_evaluate()
                    wandb.log({"actor_loss": actor_loss,
                               "bc_loss": bc_loss,
                               "ebm_loss": ebm_loss,
                               "Q_pi_loss": critic_loss_pi,
                               "Q_pi_mean": Q_pi_mean,
                               "energy": energy,
                               "energy_mean": energy_mean,
                               "evaluate_rewards": evaluate_reward,
                               # "dual_lmbda": self.lmbda.mean().item(),
                               "dual_alpha": alpha,
                               "it_steps": total_it
                               })
            self.total_it += 1
            # if total_it % 100000 == 0:
            #     self.save_parameters()

    def train_Q_pi(self, state, action, next_state, reward, not_done):
        """
        train the Q function of the learned policy: \pi
        """
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            # noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            # next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            next_action = self.actor_target.get_action(next_state)
            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q

        # action.requires_grad = True
        # energy = self.ebm.energy(state, action)
        Q1, Q2 = self.critic_net(state, action)

        # de_da = torch.autograd.grad(energy.mean(), action, create_graph=True)
        # dq1_da = torch.autograd.grad(Q1.mean(), action, create_graph=True)
        # dq2_da = torch.autograd.grad(Q2.mean(), action, create_graph=True)
        # de_da = de_da[0]
        # dq1_da = dq1_da[0]
        # dq2_da = dq2_da[0]

        # Q1_mean = Q1.mean()
        # Q2_mean = Q2.mean()
        # dq1_da = torch.autograd.grad(Q1_mean, action, create_graph=True)
        # dq2_da = torch.autograd.grad(Q2_mean, action, create_graph=True)
        # dq1_da = dq1_da[0].mean()
        # dq2_da = dq2_da[0].mean()

        # Critic loss
        critic_loss = nn.MSELoss()(Q1, target_Q) + nn.MSELoss()(Q2, target_Q)  # + nn.MSELoss()(de_da, dq1_da) + nn.MSELoss()(de_da, dq2_da)

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
        action_pi = self.actor_net.get_action(state)
        Q_pi = self.critic_net.Q1(state, action_pi)

        lmbda = self.alpha / Q_pi.abs().mean().detach()

        bc_loss = nn.MSELoss()(action_pi, action)

        energy = self.ebm.energy(state, action_pi)
        energy_loss = (energy - self.ebm.energy(state, action).detach()).mean()
        energy_mean = energy.mean()

        # no_warm_up
        actor_loss = -lmbda * Q_pi.mean() + energy_loss
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
               Q_pi.mean().cpu().detach().numpy().item(), \
               energy_loss.cpu().detach().numpy().item(), \
               energy_mean.cpu().detach().numpy().item()

    def train_actor_with_auto_alpha(self, state, action):
        """
               train the learned policy
               """
        # Actor loss
        action_pi, _, a_distribution = self.actor_net(state)
        Q_pi = self.critic_net.Q1(state, action_pi)

        lmbda = self.alpha / Q_pi.abs().mean().detach()

        bc_loss = nn.MSELoss()(action_pi, action)

        energy = self.ebm.energy(state, action_pi)
        energy_diff = (energy - self.ebm.energy(state, action).detach()).mean()

        self.auto_alpha_update(energy_diff)

        alpha = torch.clip(self.auto_alpha, self.lmbda_min, self.lmbda_max)
        energy_loss = alpha * (energy_diff - self.lmbda_thre)
        energy_mean = energy.mean()

        entropy_loss = 0.005 * a_distribution.entropy().mean()
        # actor_loss = -lmbda * Q_pi.mean() - entropy_loss
        actor_loss = -lmbda * Q_pi.mean() + energy_loss  # - entropy_loss
        # actor_loss = bc_loss
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
               Q_pi.mean().cpu().detach().numpy().item(), \
               energy_diff.cpu().detach().numpy().item(), \
               energy_mean.cpu().detach().numpy().item(), \
               alpha.cpu().detach().numpy().item()

    def train_actor_with_fisher(self, state, action):
        """
               train the learned policy
               """
        # Actor loss
        action_pi, _, a_distribution = self.actor_net(state)
        Q_pi = self.critic_net.Q1(state, action_pi)

        dq_da = torch.autograd.grad(Q_pi.mean(), action_pi, create_graph=True)
        dq_da = dq_da[0]

        lmbda = self.alpha / Q_pi.abs().mean().detach()

        bc_loss = nn.MSELoss()(action_pi, action)

        energy = self.ebm.energy(state, action_pi)
        energy_mean = energy.mean()
        de_da = torch.autograd.grad(energy_mean, action_pi, create_graph=True)
        de_da = de_da[0]

        fisher_loss = nn.MSELoss()(dq_da, de_da)

        energy_diff = (energy - self.ebm.energy(state, action).detach()).mean()

        self.auto_fisher_alpha_update(fisher_loss)

        alpha = torch.clip(self.auto_alpha, self.lmbda_min, self.lmbda_max)
        energy_loss = alpha * (energy_diff - self.lmbda_thre)
        energy_mean = energy.mean()

        entropy_loss = 0.005 * a_distribution.entropy().mean()
        actor_loss = -lmbda * Q_pi.mean() + alpha * fisher_loss - entropy_loss
        # actor_loss = bc_loss
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
               Q_pi.mean().cpu().detach().numpy().item(), \
               energy_diff.cpu().detach().numpy().item(), \
               energy_mean.cpu().detach().numpy().item(), \
               alpha.cpu().detach().numpy().item()

    def auto_fisher_alpha_update(self, fisher_loss):
        aaa = fisher_loss - 0.001
        alpha_loss = - self.auto_alpha * aaa
        self.auto_alpha_optim.zero_grad()
        alpha_loss.backward(retain_graph=True)  # ##########################  here has a bug !!!
        self.auto_alpha_optim.step()

    def auto_alpha_update(self, energy_diff):
        aaa = energy_diff - self.lmbda_thre
        alpha_loss = - self.auto_alpha * aaa
        self.auto_alpha_optim.zero_grad()
        alpha_loss.backward(retain_graph=True)  # ##########################  here has a bug !!!
        self.auto_alpha_optim.step()
        # self.auto_alpha = torch.clamp(self.auto_alpha, lmbda_min, lmbda_max)

    def train_Energy(self, state, action):
        action.requires_grad = True
        probability = self.ebm(state, action)
        energy = self.ebm.energy(state, action)
        energy_mean = energy.mean()
        de_da = torch.autograd.grad(energy_mean, action, create_graph=True)
        de_da = de_da[0].mean()

        ebm_loss = torch.sum(-torch.log(probability + 1e-5)) + de_da * 0.2
        self.ebm_optim.zero_grad()
        ebm_loss.backward()
        self.ebm_optim.step()
        return ebm_loss.cpu().detach().numpy().item()

    def train_strong_Energy(self, state, action):
        action_policy = self.actor_net.get_action(state).detach()
        probability = self.ebm.Strong_contrastive(state, action, action_policy)
        ebm_loss = torch.sum(-torch.log(probability + 1e-5))
        self.ebm_optim.zero_grad()
        ebm_loss.backward()
        self.ebm_optim.step()
        return ebm_loss.cpu().detach().numpy().item()

    def rollout_evaluate(self):
        """
        policy evaluation function
        :return: the evaluation result
        """
        ep_rews = 0.
        state = self.env.reset()
        while True:
            # state = (state - self.s_mean) / (self.s_std + 1e-5)
            state = state.squeeze()
            action = self.actor_net.get_action(state).cpu().detach().numpy()
            state, reward, done, _ = self.env.step(action)
            ep_rews += reward
            # self.env.render()
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
