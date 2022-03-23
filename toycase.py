import copy
import matplotlib.pyplot as plt
import numpy as np
from Network.Actor_Critic_net import Actor_deterministic, Double_Critic, V_critic, EBM
from Sample_Dataset.Sample_from_dataset import ReplayBuffer
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

plot_type = 'circle'

num_state, num_action, num_hidden, device = 1, 1, 32, 'cpu'
alpha = 2.5
gamma = 0.95
tau = 0.005
policy_noise = 0.2
policy_freq = 2
explore_freq = 10
start_steps = 10000
noise_clip = 0.5
evaluate_freq = 100
batch_size = 32
device = device
max_action = 1.
state_scale = 10.
action_data = 0.4
total_it = 0
max_steps = 50

if plot_type == 'circle':
    initial_state = 0.9
elif plot_type == 'w_b':
    initial_state = 0.5
else:
    initial_state = 0.7

actor_net = Actor_deterministic(num_state, num_action, num_hidden, device, max_action=max_action).float().to(device)
actor_target = copy.deepcopy(actor_net)
actor_optim = torch.optim.Adam(actor_net.parameters(), lr=3e-4)

critic_net = Double_Critic(num_state, num_action, num_hidden, device).float().to(device)
critic_target = copy.deepcopy(critic_net)
critic_optim = torch.optim.Adam(critic_net.parameters(), lr=3e-4)

critic_net_miu = Double_Critic(num_state, num_action, num_hidden, device).float().to(device)
critic_target_miu = copy.deepcopy(critic_net)
critic_optim_miu = torch.optim.Adam(critic_net_miu.parameters(), lr=3e-4)

replay_buffer = ReplayBuffer(state_dim=num_state, action_dim=num_action, device=device)


def positive_policy():
    action = action_data + (np.random.rand() - 0.5) * 0.1
    return action


def negative_policy():
    action = -action_data + (np.random.rand() - 0.5) * 0.1
    return action


def positive_circle_policy_w_b(state):
    state = copy.deepcopy(state)
    state = state/state_scale

    if state <= 0.5:
        action = (np.squeeze(np.sqrt(1. * 1. - np.square(state)), axis=0) + (np.random.rand() - 0.5) * 0.1) * 0.5
    else:
        action = -np.random.rand() * 0.4

    return action


def negative_circle_policy_w_b(state):
    state = copy.deepcopy(state)
    state = state/state_scale

    if state >= -0.5:
        action = (-np.squeeze(np.sqrt(1. * 1. - np.square(state)), axis=0) + (np.random.rand() - 0.5) * 0.1) * 0.5
    else:
        action = np.random.rand() * 0.4

    return action


def positive_circle_policy(state):
    state = copy.deepcopy(state)
    state = state/state_scale
    action = (np.squeeze(np.sqrt(1. * 1. - np.square(state)), axis=0) + (np.random.rand() - 0.5) * 0.1) * 0.5
    return action


def negative_circle_policy(state):
    state = copy.deepcopy(state)
    state = state/state_scale
    action = (-np.squeeze(np.sqrt(1. * 1. - np.square(state)), axis=0) + (np.random.rand() - 0.5) * 0.1) * 0.5
    return action


class random_wave_onedim(object):
    def __init__(self):
        super(random_wave_onedim, self).__init__()
        self.state = 0.
        self.action = 0.
        self.destination = 1.
        self.state_scale = state_scale
        self.max_action = max_action
        self.i = 0
        self.max_steps = max_steps
        self.initial_state = initial_state

        self.s_buffer = []
        self.a_buffer = []
        self.next_s_buffer = []
        self.next_a_buffer = []
        self.r_buffer = []
        self.d_buffer = []

    def reward_fn(self, next_state, action):
        reward = (400 - (np.square(next_state - self.destination)))/400
        # reward = (np.abs(next_state - self.destination))
        # reward = np.square(action) * 10000
        # if np.linalg.norm(next_state - self.destination) <= 0.1:
        #     reward = 1.
        # else:
        #     reward = 0.
        # if np.abs(action) <= 0.01:
        #     reward += 5.
        # if np.abs(next_state) <= 0.1 and np.abs(action) <= 0.005:
        #     reward += 100.
        return reward

    def reset(self):
        self.state = (np.random.rand(1) - 0.5) * 2 * self.state_scale
        return self.state

    def reset_to_positive(self):
        self.state = np.ones(1) * self.state_scale * self.initial_state
        return self.state

    def reset_to_negative(self):
        self.state = - np.ones(1) * self.state_scale * self.initial_state
        return self.state

    def action_rescale(self, action):
        return np.clip(action, -self.max_action, self.max_action)

    def add(self, s, a, r, next_s, d):
        self.s_buffer.append(s)
        self.a_buffer.append(a)
        self.r_buffer.append(r)
        self.next_s_buffer.append(next_s)
        self.d_buffer.append(d)

    def buffer_to_dataset(self):
        return {
            'observations': np.array(self.s_buffer),
            'actions': np.array(self.a_buffer),
            'next_observations': np.array(self.next_s_buffer),
            'rewards': np.array(self.r_buffer),
            'terminals': np.array(self.d_buffer),
        }

    def mc_q(self, state, action, actor):
        mc_q = 0.
        state = copy.deepcopy(state)
        action = copy.deepcopy(action)

        action = torch.unsqueeze(action, dim=0).cpu().numpy()
        self.state = torch.unsqueeze(state, dim=0).cpu().numpy()
        next_s, reward, done = self.step(action)
        mc_q = reward + gamma * mc_q
        state = next_s
        while not done:
            action = actor(state).detach().cpu().numpy()
            next_s, reward, done = self.step(action)
            mc_q = reward + gamma * mc_q
            state = next_s
        return mc_q

    def step(self, action):
        if self.i >= self.max_steps - 1:
            done = True
            self.i = 0
        else:
            done = False
        action = self.action_rescale(action)
        action = copy.deepcopy(action)
        state = copy.deepcopy(self.state)
        self.state += action

        self.state = np.clip(self.state, -self.state_scale, self.state_scale)
        next_s = copy.deepcopy(self.state)
        reward = self.reward_fn(self.state, action)

        self.add(state, action, reward, next_s, done)
        self.i += 1

        return self.state, reward, done


def train_Q_pi(state, action, next_state, reward, not_done):
    """
    train the Q function of the learned policy: \pi
    """
    action = action.unsqueeze(dim=1)
    with torch.no_grad():
        # Select action according to policy and add clipped noise
        noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
        next_action = (actor_target(next_state) + noise).clamp(-max_action, max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + gamma * target_Q

    Q1, Q2 = critic_net(state, action)

    # Critic loss
    critic_loss = nn.MSELoss()(Q1, target_Q) + nn.MSELoss()(Q2, target_Q)

    # Optimize Critic
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()
    return critic_loss.cpu().detach().numpy().item()


def train_actor(state, action):
    """
    train the learned policy
    """
    # Actor loss
    action_pi = actor_net(state)
    Q_pi = critic_net.Q1(state, action_pi)

    lmbda = alpha / Q_pi.abs().mean().detach()

    bc_loss = nn.MSELoss()(action_pi, action)

    # distance = ebm.energy(state, action_pi).mean()
    actor_loss = -lmbda * Q_pi.mean() + bc_loss

    # Optimize Actor
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

    # update the frozen target models
    for param, target_param in zip(critic_net.parameters(), critic_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    for param, target_param in zip(actor_net.parameters(), actor_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    return actor_loss.cpu().detach().numpy().item(), \
           bc_loss.cpu().detach().numpy().item(), \
           Q_pi.mean().cpu().detach().numpy().item()


def rollout_evaluate(env):
    """
    policy evaluation function
    :return: the evaluation result
    """
    state = env.reset()
    ep_rews = 0.

    while True:
        action = actor_net(state).cpu().detach().numpy()
        next_state, reward, done = env.step(action)
        ep_rews += reward
        state = next_state
        if done:
            break

    print('reward:', ep_rews)
    return ep_rews


def main():
    wandb.init(project="toycase", entity="facebear")
    # plt
    s_num = 20
    a_num = 20
    s = torch.linspace(-state_scale, state_scale, s_num)
    a = torch.linspace(-max_action, max_action, a_num)
    s, a = torch.meshgrid(s, a)
    s_input = torch.unsqueeze(torch.flatten(s), dim=1)
    a_output = torch.unsqueeze(torch.flatten(a), dim=1)
    q = torch.ones_like(s)
    mc = torch.ones_like(s)
    fig, ax = plt.subplots(1, 1)

    # collect data
    env = random_wave_onedim()
    for i in range(4):
        if i >= 2:
            state, done = env.reset_to_positive(), False
        else:
            state, done = env.reset_to_negative(), False

        while not done:
            if i >= 2:
                if plot_type == 'circle':
                    action = negative_circle_policy(state)
                elif plot_type == 'w_b':
                    action = negative_circle_policy_w_b(state)
                else:
                    action = negative_policy()
            else:
                if plot_type == 'circle':
                    action = positive_circle_policy(state)
                elif plot_type == 'w_b':
                    action = positive_circle_policy_w_b(state)
                else:
                    action = positive_policy()

            next_state, _, done = env.step(action)
            state = next_state

    dataset = env.buffer_to_dataset()
    xxx = dataset['observations']
    yyy = dataset['actions']
    xxx = np.squeeze(xxx, axis=1)

    replay_buffer.convert_D4RL(dataset, scale_rewards=False, scale_state=None)

    # learn and plt
    for total_it in tqdm(range(int(2e+4))):

        # sample data
        state, action, next_state, next_action, reward, not_done = replay_buffer.sample(batch_size)

        # update Critic
        critic_loss_pi = train_Q_pi(state, action, next_state, reward, not_done)

        # delayed policy update
        if total_it % policy_freq == 0:
            actor_loss, bc_loss, Q_pi_mean = train_actor(state, action)

            if total_it % 1000 == 0:
                evaluate_reward = rollout_evaluate(env)
                wandb.log({"actor_loss": actor_loss,
                           "bc_loss": bc_loss,
                           "Q_pi_loss": critic_loss_pi,
                           "Q_pi_mean": Q_pi_mean,
                           "evaluate_rewards": evaluate_reward,
                           "it_steps": total_it,
                           })
    # plt
    with torch.no_grad():
        plt.cla()
        z = torch.squeeze(critic_net.Q1(s_input, a_output), dim=1)

        for t in range(s_num):
            for j in range(a_num):
                q[t, j] = z[t * s_num + j]
                temp = 0.
                for _ in range(3):
                    temp += torch.tensor(env.mc_q(s[t, j], a[t, j], actor_net))
                mc[t, j] = temp / 3
                q[t, j] = q[t, j] - mc[t, j]

        aaa = q.min(1)[0].unsqueeze(dim=1).repeat(1, num_action)
        bbb = torch.abs(q - aaa)
        Q_res = ax.contourf(s, a, bbb.numpy(), 100, cmap=plt.get_cmap('rainbow'))

        plt.pause(0.01)

        plt.scatter(xxx, yyy, c='white')
        ax.set_title('||Q - Q^head||^2_2 surface')
        ax.set_xlabel('s')
        ax.set_ylabel('a')

    fig.colorbar(Q_res)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
