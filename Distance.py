import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import math
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import cluster, datasets, mixture
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.2, noise=0)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)

x_circles = torch.unsqueeze(torch.Tensor(noisy_circles[0][:, 0]), dim=1)
y_circles = torch.unsqueeze(torch.Tensor(noisy_circles[0][:, 1]), dim=1)

x_moons = torch.unsqueeze(torch.Tensor(noisy_moons[0][:, 0]), dim=1)
y_moons = torch.unsqueeze(torch.Tensor(noisy_moons[0][:, 1]), dim=1)

x_blobs = torch.unsqueeze(torch.Tensor(blobs[0][:, 0]), dim=1)
y_blobs = torch.unsqueeze(torch.Tensor(blobs[0][:, 1]), dim=1)

x = torch.unsqueeze(torch.linspace(-1, 1, 1500), dim=1)
y = x * x  # + 0.2 * torch.rand(x.size())


class EBM(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, device, batch_size, negative_samples, negative_policy=10):
        super(EBM, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.num_action = num_action
        self.num_state = num_state
        self.energy_scale = torch.tensor(50.)
        self.negative_policy = negative_policy
        self.negative_samples = negative_samples
        self.negative_samples_w_policy = int(negative_samples / 2 + negative_policy)
        self.fc1 = nn.Linear(num_state + num_action, num_hidden)
        # self.fc1 = nn.utils.spectral_norm(self.fc1)

        self.fc2 = nn.Linear(num_hidden, num_hidden)
        # self.fc2 = nn.utils.spectral_norm(self.fc2)

        self.fc3 = nn.Linear(num_hidden, num_hidden)
        # self.fc3 = nn.utils.spectral_norm(self.fc3)

        self.fc4 = nn.Linear(num_hidden, 1)
        # self.fc4 = nn.utils.spectral_norm(self.fc4)

    def energy(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)
        sa = torch.cat([x, y], dim=1)
        x = F.relu(self.fc1(sa))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        energy = self.fc4(x)
        energy = F.softplus(energy)
        return energy

    def distance(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)

        state = x.unsqueeze(0).repeat(self.negative_samples, 1, 1)
        state = state.view(self.batch_size * self.negative_samples, self.num_state)

        action = y.unsqueeze(0).repeat(self.negative_samples, 1, 1)
        action = action.view(self.batch_size * self.negative_samples, self.num_action)

        noise_action = ((torch.rand([self.batch_size * self.negative_samples, self.num_action]) - 0.5) * 3).to(
            self.device)

        noise2 = (torch.randn([self.batch_size * self.negative_samples, self.num_action])).to(self.device)

        noise_action2 = noise2 + action

        noise_diff = noise_action - action
        norm = torch.norm(noise_diff, dim=1, keepdim=True)
        # norm2 = torch.norm(noise2, dim=1, keepdim=True)
        norm2 = torch.zeros_like(norm)

        output = self.energy(state, noise_action)
        output2 = self.energy(state, action)
        label = norm
        return output, label, output2, norm2

    def forward(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)

        state = x.unsqueeze(0).repeat(self.negative_samples, 1, 1)
        state = state.view(self.batch_size * self.negative_samples, self.num_state)

        action = y.unsqueeze(0).repeat(self.negative_samples, 1, 1)
        action = action.view(self.batch_size * self.negative_samples, self.num_action)

        noise = (torch.randn([self.batch_size * self.negative_samples, self.num_action])).to(self.device)
        norm = torch.norm(noise, dim=1, keepdim=True)

        noise_action = noise + action
        output = self.energy(state, noise_action)
        label = F.softplus(norm) * 100
        return output, label


class EBM2(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, device, batch_size, negative_samples, negative_policy=10):
        super(EBM2, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.num_action = num_action
        self.num_state = num_state
        self.energy_scale = torch.tensor(50.)
        self.negative_policy = negative_policy
        self.negative_samples = negative_samples
        self.negative_samples_w_policy = int(negative_samples / 2 + negative_policy)
        self.fc1 = nn.Linear(num_state, num_hidden)
        self.fc2 = nn.Linear(num_action, num_hidden)
        # self.fc1 = nn.utils.spectral_norm(self.fc1)

        self.fc3 = nn.Linear(num_hidden, num_hidden)
        # self.fc2 = nn.utils.spectral_norm(self.fc2)

        self.fc4 = nn.Linear(num_hidden, num_hidden)
        # self.fc3 = nn.utils.spectral_norm(self.fc3)

        self.fc5 = nn.Linear(num_hidden, num_hidden)
        # self.fc2 = nn.utils.spectral_norm(self.fc2)

        self.fc6 = nn.Linear(num_hidden, num_hidden)

        self.fc7 = nn.Linear(num_hidden + num_hidden, 1)
        # self.fc4 = nn.utils.spectral_norm(self.fc4)

    def energy(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        y = F.relu(self.fc2(y))
        y = F.relu(self.fc5(y))
        y = F.relu(self.fc6(y))

        xx = torch.cat([x, y], dim=1)
        energy = self.fc7(xx)
        energy = F.softplus(energy)
        return energy

    def distance(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)

        noise_scale = (torch.randn([self.batch_size, 1])).to(self.device)
        noise = (torch.rand([self.batch_size, self.num_action]) - 0.5).to(self.device)
        norm = torch.norm(noise, dim=1, keepdim=True)
        noise = noise / norm
        noise = noise * noise_scale

        # noise = torch.cat([noise_1, noise_2], dim=0)
        noise_action = noise + y
        output = self.energy(x, noise_action)
        label = noise_scale
        return output, label

    def forward(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)

        state = x.unsqueeze(0).repeat(self.negative_samples, 1, 1)
        state = state.view(self.batch_size * self.negative_samples, self.num_state)

        action = y.unsqueeze(0).repeat(self.negative_samples, 1, 1)
        action = action.view(self.batch_size * self.negative_samples, self.num_action)

        noise = (torch.randn([self.batch_size * self.negative_samples, self.num_action])).to(self.device)
        norm = torch.norm(noise, dim=1, keepdim=True)

        noise_action = noise + action
        output = self.energy(state, noise_action)
        label = F.softplus(norm - 1) * 20
        return output, label


model = EBM(batch_size=1500, num_state=1, num_action=1, num_hidden=256, device='cuda', negative_samples=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(1-5e-4))
# schedule = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.5, total_iters=5)
# schedule = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=6000)

# plot
x1 = torch.linspace(-4, 4, 100)
y1 = torch.linspace(-4, 4, 100)
x1, y1 = torch.meshgrid(x1, y1)

input = torch.unsqueeze(torch.flatten(x1), dim=1).to(device)
output = torch.unsqueeze(torch.flatten(y1), dim=1).to(device)

fig = plt.figure()
ax = Axes3D(fig)

# train and plot
pbar = tqdm(range(20000))
for i in pbar:
    x_input, y_input = x_circles.to(device), y_circles.to(device)
    predict, label, predict2, label2 = model.distance(x_input, y_input)
    y_input.requires_grad = True
    de_da = torch.autograd.grad(model.energy(x_input, y_input).sum(), y_input, create_graph=True)
    de_da = de_da[0].abs()
    de_da = de_da.sum()

    # loss = torch.sum(-torch.log(predict))# + de_da
    loss = nn.MSELoss()(predict, label) + nn.MSELoss()(predict2, label2) * 0.1
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # schedule.step()

    # optimizer2.step()

    if i % 1000 == 0:
        with torch.no_grad():
            # print(schedule.get_last_lr())
            plt.cla()
            z = torch.squeeze(model.energy(input, output), dim=1)
            # zz = torch.squeeze(model2.energy(input, output), dim=1)
            z1 = torch.ones_like(x1)
            for t in range(100):
                for j in range(100):
                    z1[t, j] = z[t * 100 + j]  # + zz[t*100 + j]
            z1 = z1.detach().numpy()
            surf = ax.plot_surface(x1, y1, z1, cmap=plt.get_cmap('rainbow'))

            plt.pause(0.01)

    pbar.set_description("Processing%s, loss=%s" % (i, loss.item()))

# surf = ax.contourf(x1, y1, z1, zdir='z', offset=-2, levels=100, cmap='rainbow')
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(elev=90, azim=0)
ax.set_title('energy surface')
ax.set_xlabel('input')
ax.set_ylabel('output')
ax.set_zlabel('energy')

plt.ioff()
plt.show()
