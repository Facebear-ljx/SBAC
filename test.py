import gym
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np
from Sample_Dataset.Sample_from_dataset import ReplayBuffer
import d4rl
import wandb
import mujoco_py
import d3rlpy


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-7


def rollout(pi='miu'):
    ep_rews = 0.  # episode reward
    # rollout one time
    for i in range(1):
        ep_rews = 0.
        state = env.reset()
        ep_t = 0  # episode length
        while True:
            ep_t += 1
            state -= s_mean
            state /= (s_std + 1e-5)
            action = bc_standard_net.get_action(np.expand_dims(state, 0))[0].cpu().detach().numpy()
            state, reward, done, _ = env.step(action)
            ep_rews += reward
            if done:
                break

    if pi == 'pi':
        wandb.log({"pi_episode_reward": ep_rews})
    else:
        wandb.log({"miu_episode_reward": ep_rews})


class BC_standard(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, device):
        super(BC_standard, self).__init__()
        self.device = device
        self.bn0 = nn.LayerNorm(num_state)
        self.fc1 = nn.Linear(num_state, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.mu_head = nn.Linear(num_hidden, num_action)
        self.sigma_head = nn.Linear(num_hidden, num_action)

        self._action_means = torch.tensor(0, dtype=torch.float32).to(self.device)
        self._action_mags = torch.tensor(1, dtype=torch.float32).to(self.device)

    def get_log_density(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)
        x = self.bn0(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        y = torch.clamp(y, -1. + EPS, 1. - EPS)
        y = torch.atanh(y)

        log_sigma = torch.clamp(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        logp_pi = a_distribution.log_prob(y).sum(axis=-1)
        logp_pi -= (2 * (np.log(2) - y - F.softplus(-2 * y))).sum(axis=1)
        logp_pi = torch.unsqueeze(logp_pi, dim=1)
        return logp_pi

    def get_action(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = self.bn0(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_sigma = self.sigma_head(x)

        log_sigma = torch.clip(log_sigma, LOG_STD_MIN, LOG_STD_MAX)
        sigma = torch.exp(log_sigma)

        a_distribution = Normal(mu, sigma)
        action = a_distribution.rsample()
        action = torch.tanh(action)
        return action


env_name = 'hopper-expert-v2'
batch_size = 256
wandb.init(project="test_bc", entity="facebear")

env = gym.make(env_name)
num_state = env.observation_space.shape[0]
num_action = env.action_space.shape[0]
dataset = env.get_dataset()
replay_buffer = ReplayBuffer(state_dim=num_state, action_dim=num_action, device='cpu')
s_mean, s_std = replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env, dataset), scale_rewards=True)

bc_standard_net = BC_standard(num_state=num_state, num_action=num_action, num_hidden=256, device='cpu').to('cpu')
bc_standard_optim = torch.optim.Adam(bc_standard_net.parameters(), lr=1e-3)

for i in range(1000000):
    # train
    bc_standard_net.train()

    state, action, _, _, _, _ = replay_buffer.sample(256)

    # state = (state - state.mean())/(state.std()+1e-8)
    action_log_prob = -1. * bc_standard_net.get_log_density(state, action)
    loss_bc = torch.mean(action_log_prob)
    bc_standard_optim.zero_grad()
    loss_bc.backward()
    bc_standard_optim.step()
    wandb.log({"bc_standard_loss": loss_bc.item()})

    # evalutate
    bc_standard_net.eval()
    if i % 5000 == 0:
        rollout(pi='miu')

