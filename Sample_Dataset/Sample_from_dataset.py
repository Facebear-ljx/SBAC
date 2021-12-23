import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cuda'):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.next_action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.state_buffer = []
        self.action_buffer = []
        self.next_state_buffer = []
        self.next_action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []

        self.device = torch.device(device)

    # 1. Offline RL add data function: add_data_to_buffer -> convert_buffer_to_numpy_dataset -> cat_new_dataset
    def add_data_to_buffer(self, state, action, reward, done):
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)

    # 2. Offline RL add data function: add_data_to_buffer -> convert_buffer_to_numpy_dataset -> cat_new_dataset
    def convert_buffer_to_numpy_dataset(self):
        return np.array(self.state_buffer), \
               np.array(self.action_buffer), \
               np.array(self.reward_buffer), \
               np.array(self.done_buffer)

    # 3. Offline RL add data function: add_data_to_buffer -> convert_buffer_to_numpy_dataset -> cat_new_dataset
    def cat_new_dataset(self, dataset):
        new_state, new_action, new_reward, new_done = self.convert_buffer_to_numpy_dataset()

        state = np.concatenate([dataset['observations'], new_state], axis=0)
        action = np.concatenate([dataset['actions'], new_action], axis=0)
        reward = np.concatenate([dataset['rewards'].reshape(-1, 1), new_reward.reshape(-1, 1)], axis=0)
        done = np.concatenate([dataset['terminals'].reshape(-1, 1), new_done.reshape(-1, 1)], axis=0)

        # free the buffer when you have converted the online sample to offline dataset
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        return {
            'observations': state,
            'actions': action,
            'rewards': reward,
            'terminals': done,
        }

    # TD3 add data function
    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # Offline and Online sample data from replay buffer function
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)    ###################################

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),  ####################################
            torch.FloatTensor(self.next_action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def split_dataset(self, env, dataset, terminate_on_end=False, ratio=10):
        """
            Returns datasets formatted for use by standard Q-learning algorithms,
            with observations, actions, next_observations, rewards, and a terminal
            flag.

            Args:
                env: An OfflineEnv object.
                dataset: An optional dataset to pass in for processing. If None,
                    the dataset will default to env.get_dataset()
                terminate_on_end (bool): Set done=True on the last timestep
                    in a trajectory. Default is False, and will discard the
                    last timestep in each trajectory.
                **kwargs: Arguments to pass to env.get_dataset().
                ratio=N: split the dataset into N peaces

            Returns:
                A dictionary containing keys:
                    observations: An N/ratio x dim_obs array of observations.
                    actions: An N/ratio x dim_action array of actions.
                    next_observations: An N/ratio x dim_obs array of next observations.
                    rewards: An N/ratio-dim float array of rewards.
                    terminals: An N/ratio-dim boolean array of "done" or episode termination flags.
            """
        N = dataset['rewards'].shape[0]
        obs_ = []
        next_obs_ = []
        action_ = []
        reward_ = []
        done_ = []

        # The newer version of the dataset adds an explicit
        # timeouts field. Keep old method for backwards compatability.
        use_timeouts = False
        if 'timeouts' in dataset:
            use_timeouts = True

        episode_step = 0
        for i in range(int(N / ratio) - 1):
            obs = dataset['observations'][i].astype(np.float32)
            new_obs = dataset['observations'][i + 1].astype(np.float32)
            action = dataset['actions'][i].astype(np.float32)
            reward = dataset['rewards'][i].astype(np.float32)
            done_bool = bool(dataset['terminals'][i])

            if use_timeouts:
                final_timestep = dataset['timeouts'][i]
            else:
                final_timestep = (episode_step == env._max_episode_steps - 1)
            if (not terminate_on_end) and final_timestep:
                # Skip this transition and don't apply terminals on the last step of an episode
                episode_step = 0
                continue
            if done_bool or final_timestep:
                episode_step = 0

            obs_.append(obs)
            next_obs_.append(new_obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done_bool)
            episode_step += 1

        return {
            'observations': np.array(obs_),
            'actions': np.array(action_),
            'next_observations': np.array(next_obs_),
            'rewards': np.array(reward_),
            'terminals': np.array(done_),
        }

    def convert_D4RL(self, dataset, scale_rewards=False, scale_state=False):
        dataset_size = len(dataset['observations'])
        dataset['terminals'] = np.squeeze(dataset['terminals'])
        dataset['rewards'] = np.squeeze(dataset['rewards'])

        nonterminal_steps, = np.where(
            np.logical_and(
                np.logical_not(dataset['terminals']),
                np.arange(dataset_size) < dataset_size - 1))

        print('Found %d non-terminal steps out of a total of %d steps.' % (
            len(nonterminal_steps), dataset_size))

        self.state = dataset['observations'][nonterminal_steps]
        self.action = dataset['actions'][nonterminal_steps]
        self.next_state = dataset['observations'][nonterminal_steps + 1]
        self.next_action = dataset['actions'][nonterminal_steps + 1]
        self.reward = dataset['rewards'][nonterminal_steps].reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'][nonterminal_steps + 1].reshape(-1, 1)
        self.size = self.state.shape[0]

        # min_max normalization
        if scale_rewards:
            r_max = np.max(self.reward)
            r_min = np.min(self.reward)
            self.reward = (self.reward - r_min) / (r_max - r_min)

        s_mean = self.state.mean(0, keepdims=True)
        s_std = self.state.std(0, keepdims=True)

        # standard normalization
        if scale_state:
            self.state = (self.state - s_mean) / (s_std + 1e-3)
            self.next_state = (self.next_state - s_mean) / (s_std + 1e-3)

        return s_mean, s_std

    def normalize_states(self, eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std
