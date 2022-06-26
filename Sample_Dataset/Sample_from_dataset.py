import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(2e6), device='cuda'):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.s0, self.s1, self.s2, self.s3, self.s4, self.s5, self.s6, self.s7 = np.zeros(1), np.zeros(1), np.zeros(
            1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
        self.a0, self.a1, self.a2, self.a3, self.a4, self.a5, self.a6, self.a7 = np.zeros(1), np.zeros(1), np.zeros(
            1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
        self.d1, self.d2, self.d3, self.d4, self.d5, self.d6, self.d7 = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(
            1), np.zeros(1), np.zeros(1), np.zeros(1)
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.next_action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.reward1 = np.zeros((max_size, 1))
        self.reward2 = np.zeros((max_size, 1))
        self.reward3 = np.zeros((max_size, 1))
        self.reward4 = np.zeros((max_size, 1))
        self.reward5 = np.zeros((max_size, 1))
        self.reward6 = np.zeros((max_size, 1))
        self.next_reward = np.zeros((max_size, 1))
        self.nn_reward = np.zeros((max_size, 1))
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
        ind = np.random.randint(0, self.size, size=batch_size)  ###################################

        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],  ####################################
            self.next_action[ind],
            self.reward[ind],
            self.not_done[ind]
        )

    def sample_multiple(self, batch_size):
        """
        used for convert_D4RL_macro
        :param batch_size:
        :return:
        """
        ind = np.random.randint(0, self.size, size=batch_size)  ###################################

        return (
            torch.FloatTensor(self.s0[ind]).to(self.device),
            torch.FloatTensor(self.s1[ind]).to(self.device),
            torch.FloatTensor(self.s2[ind]).to(self.device),
            torch.FloatTensor(self.s3[ind]).to(self.device),
            torch.FloatTensor(self.s4[ind]).to(self.device),
            torch.FloatTensor(self.s5[ind]).to(self.device),
            torch.FloatTensor(self.s6[ind]).to(self.device),
            torch.FloatTensor(self.s7[ind]).to(self.device),

            torch.FloatTensor(self.a0[ind]).to(self.device),
            torch.FloatTensor(self.a1[ind]).to(self.device),
            torch.FloatTensor(self.a2[ind]).to(self.device),
            torch.FloatTensor(self.a3[ind]).to(self.device),
            torch.FloatTensor(self.a4[ind]).to(self.device),
            torch.FloatTensor(self.a5[ind]).to(self.device),
            torch.FloatTensor(self.a6[ind]).to(self.device),
            torch.FloatTensor(self.a7[ind]).to(self.device),

            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.reward1[ind]).to(self.device),
            torch.FloatTensor(self.reward2[ind]).to(self.device),
            torch.FloatTensor(self.reward3[ind]).to(self.device),
            torch.FloatTensor(self.reward4[ind]).to(self.device),
            torch.FloatTensor(self.reward5[ind]).to(self.device),
            torch.FloatTensor(self.reward6[ind]).to(self.device),
            # torch.FloatTensor(self.d1[ind]).to(self.device),
            # torch.FloatTensor(self.d2[ind]).to(self.device),
            # torch.FloatTensor(self.d3[ind]).to(self.device),
            torch.FloatTensor(self.d7[ind]).to(self.device),
        )

    def sample_lambda(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)  ###################################

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),  ####################################
            torch.FloatTensor(self.next_action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def split_dataset(self, env, dataset, terminate_on_end=False, ratio=10, toycase=False, env_name=None):
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
            if 'large' in env_name:
                if (0 <= dataset['observations'][i, 0] <= 0 and 15 <= dataset['observations'][i, 1] <= 18) \
                        and (10.5 <= dataset['observations'][i, 0] <= 21 and 7 <= dataset['observations'][i, 1] <= 9) \
                        and (0 <= dataset['observations'][i, 0] <= 0 and 6.5 <= dataset['observations'][i, 1] <= 9.5) \
                        and (19 <= dataset['observations'][i, 0] <= 29.5 and 15 <= dataset['observations'][i, 1] <= 17) \
                        and toycase:
                    # print('find a point')
                    continue
            # elif 'antmaze-medium' in env_name:
            #     if (11.5 <= dataset['observations'][i, 0] <= 20.5 and 11 <= dataset['observations'][i, 1] <= 13) \
            #             or (4 <= dataset['observations'][i, 0] <= 13 and 7 <= dataset['observations'][i, 1] <= 9) \
            #             and toycase:
                    # print(dataset['observations'][i, 0])
                    # print(dataset['observations'][i, 1])
                    # print(np.logical_and(10 <= dataset['observations'][i, 0] <= 15, 5 <= dataset['observations'][i, 1] <= 20))
                    # print('find a point')
                    # continue
            elif 'umaze' in env_name:
                if 6 <= dataset['observations'][i, 0] <= 10 \
                        and 2 <= dataset['observations'][i, 1] <= 6 \
                        and toycase:
                    # print('find a point')
                    continue
            # elif 'hopper' in env_name:
            #     if 1.5 <= dataset['infos/qvel'][i, 0] <= 2.5:
            #         continue
            # elif 'halfcheetah' in env_name:
            #     if 1.5 <= dataset['infos/qvel'][i, 0] <= 4:
            #         continue

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

    def convert_D4RL(self, dataset, scale_rewards=False, scale_state=False, scale_action=False, norm_reward=False):
        """
        convert the D4RL dataset into numpy ndarray, you can select whether normalize the rewards and states
        :param scale_action:
        :param dataset: d4rl dataset, usually comes from env.get_dataset or replay_buffer.split_dataset
        :param scale_rewards: whether scale the reward to [0, 1]
        :param scale_state: whether scale the state to standard gaussian distribution ~ N(0, 1)
        :return: the mean and standard deviation of states
        """
        dataset_size = len(dataset['observations'])
        dataset['terminals'] = np.squeeze(dataset['terminals'])
        dataset['rewards'] = np.squeeze(dataset['rewards'])

        nonterminal_steps, = np.where(
            np.logical_and(
                np.logical_not(dataset['terminals']),
                np.arange(dataset_size) < dataset_size - 1))
        print('Found %d non-terminal steps out of a total of %d steps.' % (
            len(nonterminal_steps), dataset_size))
        self.state = torch.FloatTensor(dataset['observations'][nonterminal_steps]).to(self.device)
        self.action = torch.FloatTensor(dataset['actions'][nonterminal_steps]).to(self.device)
        self.next_state = torch.FloatTensor(dataset['observations'][nonterminal_steps + 1]).to(self.device)  ####################################
        self.next_action = torch.FloatTensor(dataset['actions'][nonterminal_steps + 1]).to(self.device)
        self.reward = torch.FloatTensor(dataset['rewards'][nonterminal_steps].reshape(-1, 1)).to(self.device)
        self.not_done = torch.FloatTensor(1. - dataset['terminals'][nonterminal_steps + 1].reshape(-1, 1)).to(self.device)
        self.size = self.state.shape[0]

        # min_max normalization
        if scale_rewards:
            self.reward -= 1.
            # r_max = np.max(self.reward)
            # r_min = np.min(self.reward)
            # self.reward = (self.reward - r_min) / (r_max - r_min)
        else:
            if norm_reward:
                r_min = self.reward.min(0, keepdims=True)[0]
                r_max = self.reward.max(0, keepdims=True)[0]
                self.reward = (self.reward - r_min) / (r_max - r_min)

        s_mean = self.state.mean(0, keepdims=True)
        s_std = self.state.std(0, keepdims=True)

        s_min = self.state.min(0, keepdims=True)
        s_max = self.state.max(0, keepdims=True)

        a_mean = self.action.mean(0, keepdims=True)
        a_std = self.action.std(0, keepdims=True)

        if scale_state == 'minmax':
            # min_max normalization
            self.state = (self.state - s_min) / (s_max - s_min)
            self.next_state = (self.next_state - s_min) / (s_max - s_min)
            if scale_action:
                self.action = (self.action - a_mean) / (a_std + 1e-3)
                self.next_action = (self.next_action - a_mean) / (a_std + 1e-3)
                return s_min.cpu().numpy(), s_max.cpu().numpy(), a_mean.cpu().numpy(), a_std.cpu().numpy()
            else:
                return s_min.cpu().numpy(), s_max.cpu().numpy()

        elif scale_state == 'standard':
            # standard normalization
            self.state = (self.state - s_mean) / (s_std + 1e-3)
            self.next_state = (self.next_state - s_mean) / (s_std + 1e-3)
            if scale_action:
                self.action = (self.action - a_mean) / (a_std + 1e-3)
                self.next_action = (self.next_action - a_mean) / (a_std + 1e-3)
                a_max = self.action.max(0, keepdims=True)
                a_min = self.action.min(0, keepdims=True)
                return s_mean.cpu().numpy(), s_std.cpu().numpy(), a_mean.cpu().numpy(), a_std.cpu().numpy(), a_max.cpu().numpy(), a_min.cpu().numpy()
            else:
                return s_mean.cpu().numpy(), s_std.cpu().numpy()

        else:
            if scale_action:
                self.action = (self.action - a_mean) / (a_std + 1e-3)
                self.next_action = (self.next_action - a_mean) / (a_std + 1e-3)
                return s_mean.cpu().numpy(), s_std.cpu().numpy(), a_mean.cpu().numpy(), a_std.cpu().numpy()
            else:
                return s_mean.cpu().numpy(), s_std.cpu().numpy()

    def convert_D4RL_iql(self, dataset, scale_rewards=False, scale_state=False, scale_action=False, norm_reward=False):
        """
        convert the D4RL dataset into numpy ndarray, you can select whether normalize the rewards and states
        :param scale_action:
        :param dataset: d4rl dataset, usually comes from env.get_dataset or replay_buffer.split_dataset
        :param scale_rewards: whether scale the reward to [0, 1]
        :param scale_state: whether scale the state to standard gaussian distribution ~ N(0, 1)
        :return: the mean and standard deviation of states
        """
        dataset_size = len(dataset['observations'])
        dataset['terminals'] = np.squeeze(dataset['terminals'])
        dataset['rewards'] = np.squeeze(dataset['rewards'])

        # nonterminal_steps, = np.where(
        #     np.logical_and(
        #         np.logical_not(dataset['terminals']),
        #         np.arange(dataset_size) < dataset_size - 1))
        print('Found non-terminal steps out of a total of %d steps.' % (dataset_size))
        self.state = torch.FloatTensor(dataset['observations']).to(self.device)
        self.action = torch.FloatTensor(dataset['actions']).to(self.device)
        self.next_state = torch.FloatTensor(dataset['next_observations']).to(self.device)  ####################################
        self.next_action = torch.FloatTensor(dataset['actions']).to(self.device)
        self.reward = torch.FloatTensor(dataset['rewards'].reshape(-1, 1)).to(self.device)
        self.not_done = torch.FloatTensor(1. - dataset['terminals'].reshape(-1, 1)).to(self.device)
        self.size = self.state.shape[0]

        # min_max normalization
        if scale_rewards:
            self.reward -= 1.
            # r_max = np.max(self.reward)
            # r_min = np.min(self.reward)
            # self.reward = (self.reward - r_min) / (r_max - r_min)
        else:
            if norm_reward:
                r_min = self.reward.min(0, keepdims=True)[0]
                r_max = self.reward.max(0, keepdims=True)[0]
                self.reward = (self.reward - r_min) / (r_max - r_min)

        s_mean = self.state.mean(0, keepdims=True)
        s_std = self.state.std(0, keepdims=True)

        s_min = self.state.min(0, keepdims=True)
        s_max = self.state.max(0, keepdims=True)

        a_mean = self.action.mean(0, keepdims=True)
        a_std = self.action.std(0, keepdims=True)

        if scale_state == 'minmax':
            # min_max normalization
            self.state = (self.state - s_min) / (s_max - s_min)
            self.next_state = (self.next_state - s_min) / (s_max - s_min)
            if scale_action:
                self.action = (self.action - a_mean) / (a_std + 1e-3)
                self.next_action = (self.next_action - a_mean) / (a_std + 1e-3)
                return s_min.cpu().numpy(), s_max.cpu().numpy(), a_mean.cpu().numpy(), a_std.cpu().numpy()
            else:
                return s_min.cpu().numpy(), s_max.cpu().numpy()

        elif scale_state == 'standard':
            # standard normalization
            self.state = (self.state - s_mean) / (s_std + 1e-3)
            self.next_state = (self.next_state - s_mean) / (s_std + 1e-3)
            if scale_action:
                self.action = (self.action - a_mean) / (a_std + 1e-3)
                self.next_action = (self.next_action - a_mean) / (a_std + 1e-3)
                a_max = self.action.max(0, keepdims=True)
                a_min = self.action.min(0, keepdims=True)
                return s_mean.cpu().numpy(), s_std.cpu().numpy(), a_mean.cpu().numpy(), a_std.cpu().numpy(), a_max.cpu().numpy(), a_min.cpu().numpy()
            else:
                return s_mean.cpu().numpy(), s_std.cpu().numpy()

        else:
            if scale_action:
                self.action = (self.action - a_mean) / (a_std + 1e-3)
                self.next_action = (self.next_action - a_mean) / (a_std + 1e-3)
                return s_mean.cpu().numpy(), s_std.cpu().numpy(), a_mean.cpu().numpy(), a_std.cpu().numpy()
            else:
                return s_mean.cpu().numpy(), s_std.cpu().numpy()


    def convert_D4RL_td_lambda(self, dataset, scale_rewards=False, scale_state=False, n=1):
        """
        convert the D4RL dataset into numpy ndarray, you can select whether normalize the rewards and states
        :param n: TD error skip steps
        :param dataset: d4rl dataset, usually comes from env.get_dataset or replay_buffer.split_dataset
        :param scale_rewards: whether scale the reward to [0, 1]
        :param scale_state: whether scale the state to standard gaussian distribution ~ N(0, 1)
        :return: the mean and standard deviation of states
        """

        dataset_size = len(dataset['observations'])
        dataset['terminals'] = np.squeeze(dataset['terminals'])
        dataset['rewards'] = np.squeeze(dataset['rewards'])

        aa = np.logical_not(dataset['terminals'])
        bb = np.arange(dataset_size) < dataset_size - n

        if n == 1:
            aa = aa
        elif n == 2:
            for i in range(len(aa) - 1):
                if not aa[i + 1]:
                    aa[i] = False
        elif n == 3:
            for i in range(len(aa) - 2):
                if not aa[i + 2]:
                    aa[i] = False
                    aa[i + 1] = False
        elif n == 4:
            for i in range(len(aa) - 3):
                if not aa[i + 3]:
                    aa[i] = False
                    aa[i + 1] = False
                    aa[i + 2] = False
        else:
            print("n too large!!!!!!!!!!!!!!!!!!!!!! need 1,2,3,4")
        nonterminal_steps, = np.where(np.logical_and(aa, bb))
        print('Found %d non-terminal steps out of a total of %d steps.' % (
            len(nonterminal_steps), dataset_size))

        self.state = dataset['observations'][nonterminal_steps]
        self.action = dataset['actions'][nonterminal_steps]
        self.next_state = dataset['observations'][nonterminal_steps + n]
        self.next_action = dataset['actions'][nonterminal_steps + n]
        self.reward = dataset['rewards'][nonterminal_steps].reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'][nonterminal_steps + n].reshape(-1, 1)  #########
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

    def convert_D4RL_macro(self, dataset, scale_rewards=False, scale_state=False, n=1):
        """
        convert the D4RL dataset into numpy ndarray, you can select whether normalize the rewards and states
        :param n: TD error skip steps
        :param dataset: d4rl dataset, usually comes from env.get_dataset or replay_buffer.split_dataset
        :param scale_rewards: whether scale the reward to [0, 1]
        :param scale_state: whether scale the state to standard gaussian distribution ~ N(0, 1)
        :return: the mean and standard deviation of states
        """

        dataset_size = len(dataset['observations'])
        dataset['terminals'] = np.squeeze(dataset['terminals'])
        dataset['rewards'] = np.squeeze(dataset['rewards'])

        aa = np.logical_not(dataset['terminals'])
        bb = np.arange(dataset_size) < dataset_size - n

        if n == 1:
            aa = aa
        elif n == 2:
            for i in range(len(aa) - 1):
                if not aa[i + 1]:
                    aa[i] = False
        elif n == 3:
            for i in range(len(aa) - 2):
                if not aa[i + 2]:
                    aa[i] = False
                    aa[i + 1] = False
        elif n == 4:
            for i in range(len(aa) - 3):
                if not aa[i + 3]:
                    aa[i] = False
                    aa[i + 1] = False
                    aa[i + 2] = False
        elif n == 7:
            for i in range(len(aa) - 6):
                if not aa[i + 6]:
                    aa[i] = False
                    aa[i + 1] = False
                    aa[i + 2] = False
                    aa[i + 3] = False
                    aa[i + 4] = False
                    aa[i + 5] = False
        else:
            print("n too large!!!!!!!!!!!!!!!!!!!!!! need 1,2,3,4")
        nonterminal_steps, = np.where(np.logical_and(aa, bb))
        print('Found %d non-terminal steps out of a total of %d steps.' % (
            len(nonterminal_steps), dataset_size))

        self.s0 = dataset['observations'][nonterminal_steps]
        self.s1 = dataset['observations'][nonterminal_steps + 1]
        self.s2 = dataset['observations'][nonterminal_steps + 2]
        self.s3 = dataset['observations'][nonterminal_steps + 3]
        self.s4 = dataset['observations'][nonterminal_steps + 4]
        self.s5 = dataset['observations'][nonterminal_steps + 5]
        self.s6 = dataset['observations'][nonterminal_steps + 6]
        self.s7 = dataset['observations'][nonterminal_steps + 7]

        self.a0 = dataset['actions'][nonterminal_steps]
        self.a1 = dataset['actions'][nonterminal_steps + 1]
        self.a2 = dataset['actions'][nonterminal_steps + 2]
        self.a3 = dataset['actions'][nonterminal_steps + 3]
        self.a4 = dataset['actions'][nonterminal_steps + 4]
        self.a5 = dataset['actions'][nonterminal_steps + 5]
        self.a6 = dataset['actions'][nonterminal_steps + 6]
        self.a7 = dataset['actions'][nonterminal_steps + 7]

        self.reward = dataset['rewards'][nonterminal_steps].reshape(-1, 1)
        self.reward1 = dataset['rewards'][nonterminal_steps + 1].reshape(-1, 1)
        self.reward2 = dataset['rewards'][nonterminal_steps + 2].reshape(-1, 1)
        self.reward3 = dataset['rewards'][nonterminal_steps + 3].reshape(-1, 1)
        self.reward4 = dataset['rewards'][nonterminal_steps + 4].reshape(-1, 1)
        self.reward5 = dataset['rewards'][nonterminal_steps + 5].reshape(-1, 1)
        self.reward6 = dataset['rewards'][nonterminal_steps + 6].reshape(-1, 1)

        self.d1 = 1. - dataset['terminals'][nonterminal_steps + 1].reshape(-1, 1)
        self.d2 = 1. - dataset['terminals'][nonterminal_steps + 2].reshape(-1, 1)
        self.d3 = 1. - dataset['terminals'][nonterminal_steps + 3].reshape(-1, 1)
        self.d4 = 1. - dataset['terminals'][nonterminal_steps + 4].reshape(-1, 1)
        self.d5 = 1. - dataset['terminals'][nonterminal_steps + 5].reshape(-1, 1)
        self.d6 = 1. - dataset['terminals'][nonterminal_steps + 6].reshape(-1, 1)
        self.d7 = 1. - dataset['terminals'][nonterminal_steps + 7].reshape(-1, 1)

        self.size = self.s0.shape[0]

        # min_max normalization
        if scale_rewards:
            r_max = np.max(self.reward)
            r_min = np.min(self.reward)
            self.reward = (self.reward - r_min) / (r_max - r_min)

        s_mean = self.s0.mean(0, keepdims=True)
        s_std = self.s0.std(0, keepdims=True)

        # standard normalization
        if scale_state:
            self.s0 = (self.s0 - s_mean) / (s_std + 1e-3)
            self.s1 = (self.s1 - s_mean) / (s_std + 1e-3)
            self.s2 = (self.s2 - s_mean) / (s_std + 1e-3)
            self.s3 = (self.s3 - s_mean) / (s_std + 1e-3)
            self.s4 = (self.s4 - s_mean) / (s_std + 1e-3)
            self.s5 = (self.s5 - s_mean) / (s_std + 1e-3)
            self.s6 = (self.s6 - s_mean) / (s_std + 1e-3)
            self.s7 = (self.s7 - s_mean) / (s_std + 1e-3)

        return s_mean, s_std

    def normalize_states(self, eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std
