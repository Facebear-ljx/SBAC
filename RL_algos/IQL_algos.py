import copy
import torch
import torch.nn as nn
import wandb
import gym
import os
import d4rl
from Sample_Dataset.Sample_from_dataset import ReplayBuffer
from Network.Actor_Critic_net import Actor, Double_Critic, V_critic
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import datetime

EPSILON = 1e-20
EXP_ADV_MAX = 100

def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class IQL:
    def __init__(self,
                 env_name,
                 num_hidden=256,
                 gamma=0.99,
                 soft_update=0.005,
                #  policy_noise=0.2,
                #  noise_clip=0.5,
                #  policy_freq=2,
                 tau=0.7,
                 beta=3.0,
                 ratio=1,
                 seed=0,
                 batch_size=256,
                 scale_state='minmax',
                 lr_actor=3e-4,
                 lr_critic=3e-4,
                 lr_v=3e-4,
                 evaluate_freq=5000,
                 evalute_episodes=10,
                 n_steps=int(1e+6),
                 device='cpu'):
        """
        Facebear's implementation of IQL (OFFLINE REINFORCEMENT LEARNING WITH IMPLICIT Q-LEARNING)
        Paper: https://arxiv.org/pdf/2110.06169.pdf

        :param env_name: your gym environment name
        :param num_hidden: the number of the units of the hidden layer of your network
        :param gamma: discounting factor of the cumulated reward
        :param soft_update: soft update
        :param tau: expectile regression threshold
        :param beta: parameters
        :param policy_freq: delayed policy update frequency
        :param alpha: the hyper-parameter in equation
        :param ratio:
        :param device:
        """
        super(IQL, self).__init__()

        # hyper-parameters
        self.gamma = gamma
        self.soft_update = soft_update
        # self.policy_noise = policy_noise
        # self.policy_freq = policy_freq
        self.evaluate_freq = evaluate_freq
        self.evaluate_episodes = evalute_episodes
        # self.noise_clip = noise_clip
        self.tau = tau
        self.beta = beta
        self.batch_size = batch_size
        self.device = device
        self.max_action = 1.
        self.scale_state = scale_state
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

        # get dataset
        self.dataset = self.env.get_dataset()
        self.replay_buffer = ReplayBuffer(state_dim=num_state, action_dim=num_action, device=device)
        self.dataset = self.replay_buffer.split_dataset(self.env, self.dataset, ratio=ratio, env_name=env_name)

        if 'antmaze' in env_name:
            scale_rewards = True
        else:
            scale_rewards = False
        self.s_mean, self.s_std = self.replay_buffer.convert_D4RL(
            self.dataset,
            scale_rewards=scale_rewards,
            scale_state=scale_state,
            norm_reward=True
        )

        # prepare the actor and critic
        self.actor_net = Actor(num_state, num_action, num_hidden, device).float().to(device)
        self.actor_target = copy.deepcopy(self.actor_net)
        self.actor_optim = torch.optim.Adam(self.actor_net.parameters(), lr=lr_actor)
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optim, n_steps)

        self.critic_net = Double_Critic(num_state, num_action, num_hidden, device).float().to(device)
        self.critic_target = copy.deepcopy(self.critic_net)
        self.critic_optim = torch.optim.Adam(self.critic_net.parameters(), lr=lr_critic)

        self.v_net = V_critic(num_state, num_hidden, device).float().to(device)
        self.v_optim = torch.optim.Adam(self.v_net.parameters(), lr=lr_v)


        # Q and Critic file location
        self.current_time = datetime.datetime.now()
        logdir_name = f"./Model/{self.env_name}/{self.current_time}+{self.seed}"
        os.makedirs(logdir_name)


    def learn(self, total_time_step=1e+6):
        """
        IQL's learning framework
        :param total_time_step: the total iteration times for training IQL
        :return: None
        """
        # self.warm_up(1e+6)

        for total_it in tqdm(range(1, int(total_time_step) + 1)):

            # sample data
            state, action, next_state, _, reward, not_done = self.replay_buffer.sample(self.batch_size)

            # update V
            v_loss, advantage, v =self.train_v(state, action)

            # update Q
            critic_loss = self.train_Q(state, action, next_state, reward, not_done)

            # policy update
            actor_loss, bc_loss = self.train_actor(state, action, advantage)
            if total_it % self.evaluate_freq == 0:
                evaluate_reward = self.rollout_evaluate()
                wandb.log({ "evaluate_rewards": evaluate_reward,
                            "v_mean": v.mean().cpu().detach().numpy().item(),
                            "v_loss": v_loss.cpu().detach().numpy().item(),
                            "actor_loss": actor_loss.cpu().detach().numpy().item(),
                            "bc_loss": bc_loss.mean().cpu().detach().numpy().item(),
                            "critic_loss": critic_loss.cpu().detach().numpy().item(),
                            "it_steps": total_it,
                            })

                # if total_it % (self.evaluate_freq * 2) == 0:
                #     self.save_parameters(evaluate_reward)
            self.total_it += 1
    
    def train_v(self, state, action):
        """
        train Value function that used to calculate the target for Q training
        """
        with torch.no_grad():
            target_q1, target_q2 = self.critic_target(state, action)
            target_q = torch.min(target_q1, target_q2)
        
        v = self.v_net(state)
        advantage = target_q - v
        v_loss = asymmetric_l2_loss(advantage, self.tau)

        # Optimize Value function
        self.v_optim.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optim.step()

        return v_loss, advantage, v

    def train_Q(self, state, action, next_state, reward, not_done):
        """
        train the Q function
        """
        with torch.no_grad():
            next_v = self.v_net(next_state)
            target_q = reward + not_done * self.gamma * next_v
        Q1, Q2 = self.critic_net(state, action)
        critic_loss = nn.MSELoss()(Q1, target_q) + nn.MSELoss()(Q2, target_q)

        # Optimize Critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 5.0)
        self.critic_optim.step()
        return critic_loss


    def train_actor(self, state, action, advantage):
        """
        train the learned policy
        """
        exp_adv = torch.exp(self.beta * advantage.detach()).clamp(max=EXP_ADV_MAX)
        # Actor loss
        log_pi = self.actor_net.get_log_density(state, action)

        bc_loss = -log_pi
        
        actor_loss = torch.mean(exp_adv * bc_loss)

        # Optimize Actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.actor_lr_schedule.step()

        # update the frozen target models
        for param, target_param in zip(self.critic_net.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.soft_update * param.data + (1 - self.soft_update) * target_param.data)

        for param, target_param in zip(self.actor_net.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.soft_update * param.data + (1 - self.soft_update) * target_param.data)

        return actor_loss, bc_loss


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
