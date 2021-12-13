# 包
try:
    from local_debug_logger import local_trace
except ImportError:
    local_trace = lambda: None

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import d4rl
import gym

# set cuda
torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# makedir
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Hyper-parameters
IMAGE_SIZE = 14
H_DIM = 12
Z_DIM = 5
NUM_EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
TRAIN_DATA_TYPE = 'medium'


def feature_normalize(data):
    a = data.clone()
    a -= data.mean(1, keepdim=True)[0]
    a /= data.std(1, keepdim=True)[0]
    return a


# VAE model
class VAE(nn.Module):
    def __init__(self, inputsize=14, h_dim=10, z_dim=2):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(inputsize, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)  # 均值 向量
        self.fc3 = nn.Linear(h_dim, z_dim)  # 方差 向量
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, inputsize)

    # 编码过程
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return self.fc5(h)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var


env = gym.make('hopper-medium-v2')
env2 = gym.make('hopper-random-v2')
env3 = gym.make('hopper-medium-expert-v2')

# load the dataset
dataset_medium = env.get_dataset()
dataset_random = env2.get_dataset()
dataset_mix = env3.get_dataset()

# concatenate state and action
state_mix, action_mix = torch.Tensor(dataset_mix['observations']), torch.Tensor(dataset_mix['actions'])
state_random, action_random = torch.Tensor(dataset_random['observations']), torch.Tensor(dataset_random['actions'])
state_medium, action_medium = torch.Tensor(dataset_medium['observations']), torch.Tensor(dataset_medium['actions'])
state_action_mix = torch.cat([state_mix, action_mix], dim=1)
state_action_random = torch.cat([state_random, action_random], dim=1)
state_action_medium = torch.cat([state_medium, action_medium], dim=1)
input_size = len(state_action_medium[1])

# Split the dataset into several pieces : the data type is Subset
len_medium = len(dataset_medium['observations'])
base_size, train_size, test_size, truth_size = int(0.0 * len_medium), int(0.9989 * len_medium), int(
    0.0001 * len_medium), int(0.001 * len_medium)

train_mixed, test_mixed, truth_mixed = torch.utils.data.random_split(
    state_action_mix, [1998806, 100, 1000])

# Medium data
base_medium, train_medium, test_medium, truth_medium = torch.utils.data.random_split(
    state_action_medium, [base_size, train_size, test_size, truth_size])

# Random data
base_random, train_random, test_random, truth_random = torch.utils.data.random_split(
    state_action_random, [base_size, train_size, test_size, truth_size])

train_data = train_medium
test_data = test_medium
truth_data = truth_medium
if TRAIN_DATA_TYPE == 'medium':
    train_data = train_medium
    test_data = test_medium
    truth_data = truth_medium
    print('medium')
elif TRAIN_DATA_TYPE == 'mixed':
    train_data = train_mixed
    test_data = test_mixed
    truth_data = truth_mixed
    print('mixed')
elif TRAIN_DATA_TYPE == 'random':
    train_data = train_random
    test_data = test_random
    truth_data = truth_random
    print('random')

# data loader
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)

# setup my vae
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
MSELoss = nn.MSELoss(reduction='sum')

for epoch in range(NUM_EPOCHS):
    for i, x in enumerate(train_loader):
        x = feature_normalize(x)
        x = x.to(device)
        x_reconst, mu, log_var = model(x)
        reconst_loss = MSELoss(x_reconst, x)
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # backward and optimize
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                  .format(epoch + 1, NUM_EPOCHS, i + 1, len(train_loader), reconst_loss.item(), kl_div.item()))

torch.save(model.state_dict(), "my_vae.pth")
print("Saved VAE Model State to my_vae.pth")
