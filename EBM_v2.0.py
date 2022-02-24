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
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.2, noise=0.0)
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
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, num_hidden)
        self.fc4 = nn.Linear(num_hidden, 1)

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
        # energy = F.softplus(energy)
        # energy = torch.clip(energy, -10.)
        energy = torch.tanh(self.fc4(x)) * 10
        return energy

    def Strong_contrastive(self, x, y, y_policy):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)
        Positive_E = -self.energy(x, y)
        Positive = torch.exp(Positive_E)

        state = x.unsqueeze(0).repeat(self.negative_samples_w_policy, 1, 1)
        state = state.view(self.batch_size * self.negative_samples_w_policy, self.num_state)

        y_policy = y_policy.unsqueeze(0).repeat(self.negative_policy, 1, 1)
        y_policy = y_policy.view(self.batch_size * self.negative_policy, self.num_action)

        noise_action_1 = ((torch.rand([self.batch_size * (self.negative_samples_w_policy - self.negative_policy),
                                       self.num_action]) - 0.5) * 2.1).to(self.device)
        noise_2 = (torch.randn([self.batch_size * self.negative_policy, self.num_action]) * 0.2).clamp(-0.5, 0.5).to(
            self.device)
        noise_action_2 = (noise_2 + y_policy).clamp(-1., 1.)

        noise_action = torch.cat([noise_action_1, noise_action_2], dim=0)

        Negative_E = -self.energy(state, noise_action)
        Negative = torch.exp(Negative_E).view(self.negative_samples_w_policy, self.batch_size, 1)
        Negative = torch.sum(Negative, dim=0, keepdim=False)

        out = Positive / (Positive + Negative)
        return out

    def forward(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.float).to(self.device)
        Positive_E = -self.energy(x, y)
        Positive = torch.exp(Positive_E)
        # Positive = Positive_E
        state = x.unsqueeze(0).repeat(self.negative_samples, 1, 1)
        state = state.view(self.batch_size * self.negative_samples, self.num_state)
        noise_action = ((torch.rand([self.batch_size * self.negative_samples, self.num_action]) - 0.5) * 2.5).to(
            self.device)  # wtf, noise scale should be larger than the action range
        # noise_action = (torch.ones([self.batch_size * self.negative_samples, self.num_action])).to(self.device)
        Negative_E = -self.energy(state, noise_action)
        # Negative = Negative_E.view(self.negative_samples, self.batch_size, 1).sum(0)
        Negative = torch.exp(Negative_E).view(self.negative_samples, self.batch_size, 1).sum(0)
        # Negative = torch.sum(Negative, dim=1, keepdim=False)
        # Negative = Negative.sum(0)

        out = Positive / (Positive + Negative)
        return out


# class EBM(nn.Module):
#     def __init__(self):
#         super(EBM, self).__init__()
#         self.liner_relu_stack = nn.Sequential(
#             nn.Linear(2, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1),
#         )
#
#     def Energy(self, x, y):
#         input = torch.cat([x, y], dim=1)
#         energy = self.liner_relu_stack(input)
#         # energy = torch.tanh(energy)*10
#         return energy
#
#     def forward(self, x, y):
#         energy = -self.Energy(x, y)
#         exp = torch.exp(energy)
#         noise = 0
#         for _ in range(20):
#             noise_y = torch.randn_like(x) * 10
#             noise_energy = -self.Energy(x, noise_y)
#             noise += torch.exp(noise_energy)
#         out = exp/(exp + noise)
#         return out


model = EBM(batch_size=1500, num_state=1, num_action=1, num_hidden=256, device='cuda', negative_samples=20).to(device)
model2 = EBM(batch_size=1500, num_state=1, num_action=1, num_hidden=256, device='cuda', negative_samples=20).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

# schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=(1-5e-4))
# schedule = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.5, total_iters=5)
schedule = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=6000)

optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
# plot
x1 = torch.linspace(-4, 4, 100)
y1 = torch.linspace(-4, 4, 100)
x1, y1 = torch.meshgrid(x1, y1)

input = torch.unsqueeze(torch.flatten(x1), dim=1).to(device)
output = torch.unsqueeze(torch.flatten(y1), dim=1).to(device)

fig = plt.figure()
ax = Axes3D(fig)

# train and plot
pbar = tqdm(range(1000000))
for i in pbar:
    x_input, y_input = x_circles.to(device), x_circles.to(device)
    # x_input2, y_input2 = x_circles.to(device), y_circles.to(device)
    predict = model(x_input, y_input)
    # predict2 = model2(x_input2, y_input2)
    y_input.requires_grad = True
    de_da = torch.autograd.grad(model.energy(x_input, y_input).sum(), y_input, create_graph=True)
    de_da = de_da[0].abs()
    de_da = de_da.sum()

    loss = torch.sum(-torch.log(predict)) + de_da
    # loss2 = torch.sum(-torch.log(predict2))
    optimizer.zero_grad()
    # optimizer2.zero_grad()
    loss.backward()
    # loss2.backward()
    optimizer.step()
    schedule.step()

    # optimizer2.step()

    if i % 200 == 0:
        with torch.no_grad():
            print(schedule.get_last_lr())
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
