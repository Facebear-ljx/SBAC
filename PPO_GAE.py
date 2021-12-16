import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import time
import gym
import argparse
from torch.distributions import MultivariateNormal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


wandb.init(project="online fine-tuning", entity="facebear")

# Parameters
parser = argparse.ArgumentParser(description='Solve the Hopper-v2 with PPO')
parser.add_argument(
    '--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument(
    '--lr', type=float, default=3e-4, metavar='lr', help='discount factor (default: 3e-4)')
parser.add_argument(
    '--gae_lambda', type=float, default=0.95, metavar='λ', help='GAE factor (default: 0.95)')
parser.add_argument(
    '--ppo_epoch', type=int, default=10, metavar='K', help='ppo_train_epoch (default: 10)')
parser.add_argument(
    '--buffer_capacity', type=int, default=2048, metavar='B', help='buffer_capacity (default: 2048)')
parser.add_argument(
    '--batch_size', type=int, default=64, metavar='BS', help='batch_size (default: 64)')
parser.add_argument(
    '--num_hidden', type=int, default=256, metavar='hidden', help='num_hidden (default: 64)')
parser.add_argument('--seed', type=int, default=10, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', default=False, help='render the environment')
parser.add_argument('--device', default='cpu', help='cuda or cpu')
args = parser.parse_args()
wandb.config.update(args)


torch.manual_seed(args.seed)
# env.seed(args.seed)
device = args.device


class Actor(nn.Module):
    def __init__(self, num_state, num_action, num_hidden):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.mu_head = nn.Linear(num_hidden, num_action)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        return mu


class Critic(nn.Module):
    def __init__(self, num_state, num_hidden):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.state_value = nn.Linear(num_hidden, 1)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.state_value(x)
        return value


class PPO:
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = args.ppo_epoch
    learning_rate = args.lr
    batch_size = args.batch_size
    buffer_capacity = args.buffer_capacity
    cov_var = torch.full(size=(3,), fill_value=0.5).to(device)
    cov_mat = torch.diag(cov_var).to(device)

    def __init__(self, env, num_state, num_action, num_hidden):
        # actor critic network setup
        super(PPO, self).__init__()
        self.actor_net = Actor(num_state, num_action, num_hidden).float().to(device)
        self.critic_net = Critic(num_state, num_hidden).float().to(device)
        self.actor_optim = optim.Adam(self.actor_net.parameters(), self.learning_rate)
        self.critic_optim = optim.Adam(self.critic_net.parameters(), self.learning_rate)
        self.env = env
        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,  # timesteps so far
            'i_so_far': 0,  # iterations so far
            'batch_lens': [],  # episodic lengths in batch
            'batch_rews': [],  # episodic returns in batch
        }

    def learn(self, total_time_step=1e+8):
        t_so_far = 0
        i_so_far = 0
        while t_so_far < total_time_step:  # ALG STEP 2

            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()  # ALG STEP 3

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far
            # Calculate advantage at k-th iteration
            V, _, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()  # ALG STEP 5
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.ppo_epoch):  # ALG STEP 6 & 7
                # Calculate V_phi and pi_theta(a_t | s_t)
                for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size,
                                          drop_last=True):
                    V, curr_log_probs, entropy_loss = self.evaluate(batch_obs[index], batch_acts[index])
                    ratios = torch.exp(curr_log_probs - batch_log_probs[index])

                    # Calculate actor losses.
                    L1 = ratios * A_k[index]
                    L2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * A_k[index]

                    actor_loss = (-torch.min(L1, L2)).mean()
                    critic_loss = nn.MSELoss()(V, batch_rtgs[index])

                    # Calculate gradients and perform backward propagation for actor network
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    self.actor_optim.step()
                    wandb.log({"actor_loss": actor_loss.item()})

                    # Calculate gradients and perform backward propagation for critic network
                    self.critic_optim.zero_grad()
                    critic_loss.backward()
                    self.critic_optim.step()
                    wandb.log({"critic_loss": critic_loss.item()})

            self._log_summary()

    def choose_action(self, s):
        mean = self.actor_net(s)
        distribution = MultivariateNormal(mean, self.cov_mat)
        action = distribution.sample()
        action_log_prob = distribution.log_prob(action)
        return action.cpu().detach().numpy(), action_log_prob.detach()

    def evaluate(self, s, a):
        value = self.critic_net(s).squeeze()
        mean = self.actor_net(s)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(a)
        entropy_loss = dist.entropy().mean()
        return value, log_probs, entropy_loss

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        t = 0
        while t < self.buffer_capacity:
            ep_rews = []
            state = self.env.reset()
            ep_t = 0
            while True:
                # self.env.render()
                ep_t += 1
                t += 1  # Increment timesteps ran this batch so far
                batch_obs.append(state)

                action, log_prob = self.choose_action(state)
                state, reward, done, _ = self.env.step(action)

                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                ep_rews.append(reward)

                # If the environment tells us the episode is terminated, break
                if done:
                    break

            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float).to(device)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(device)
        batch_rtgs = self.compute_return(batch_rews).to(device)  # ALG STEP 4

        avg_ep_lens = np.mean(batch_lens)
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in batch_rews])

        wandb.log({"episode_reward": avg_ep_rews.item()})
        wandb.log({"episode_lens": avg_ep_lens.item()})
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_return(self, batch_rews):
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0  # The discounted reward so far
            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * args.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(device)

        return batch_rtgs

    def _log_summary(self):
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []


def main():
    env_name = 'Hopper-v2'
    env = gym.make(env_name)
    num_state = env.observation_space.shape[0]
    num_action = env.action_space.shape[0]
    num_hidden = args.num_hidden
    agent = PPO(env, num_state, num_action, num_hidden)
    agent.learn(total_time_step=200_000_000)
    wandb.watch(agent)


if __name__ == '__main__':
    main()
