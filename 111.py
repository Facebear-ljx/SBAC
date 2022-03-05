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

device = "cpu" if torch.cuda.is_available() else "cpu"
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
# + 0.2 * torch.rand(x.size())


x = torch.rand([1500, 1]) - 0.7
x = torch.heaviside(x, torch.tensor([0.]))
# x = torch.zeros([1500, 1])
x_n = torch.randn([1500, 1])*0.2
x = x_n+x


class EBM(nn.Module):
    def __init__(self, num_state, num_action, num_hidden, device, batch_size, negative_samples, negative_policy=10):
        super(EBM, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.num_action = num_action
        self.num_state = num_state
        self.energy_scale = torch.tensor(50.)
        self.negative_samples = negative_samples
        self.fc1 = nn.Linear(num_state, num_hidden)

        self.fc2 = nn.Linear(num_hidden, num_hidden)

        self.fc3 = nn.Linear(num_hidden, 1)
        self.fc3 = nn.utils.spectral_norm(self.fc3)

    def energy(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        energy = self.fc3(x)
        energy = F.softplus(energy)
        return energy

    def distance(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float).to(self.device)

        x = x.unsqueeze(0).repeat(self.negative_samples, 1, 1)
        x = x.view(self.batch_size * self.negative_samples, self.num_state)

        noise_x = ((torch.rand([self.batch_size * self.negative_samples, 1]) - 0.5) * 3).to(self.device)

        noise2 = (torch.randn([self.batch_size * self.negative_samples, self.num_action])).to(self.device)

        noise_x2 = noise2 + x

        noise_diff = noise_x - x
        norm = torch.norm(noise_diff, dim=1, keepdim=True)
        # norm2 = torch.norm(noise2, dim=1, keepdim=True)
        norm2 = torch.zeros_like(x) #+ torch.rand([15000, 1])*0.1

        output = self.energy(noise_x)
        output2 = self.energy(x)

        label = norm
        return output, label, output2, norm2


model = EBM(batch_size=1500, num_state=1, num_action=1, num_hidden=256, device='cpu', negative_samples=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# plot
x1 = torch.linspace(-4, 4, 100).unsqueeze(1)

fig = plt.figure()

# train and plot
pbar = tqdm(range(10000))
for i in pbar:
    x_input = x.to(device)
    predict, label, predict2, label2 = model.distance(x_input)

    # loss = torch.sum(-torch.log(predict))# + de_da
    loss = nn.MSELoss()(predict, label) + nn.MSELoss()(predict2, label2) * 0.01
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 5 == 0:
        with torch.no_grad():
            plt.cla()
            energy = model.energy(x1).squeeze(1).numpy()
            plt.plot(x1.squeeze().numpy(), energy, '-')
            plt.pause(0.01)

    pbar.set_description("Processing%s, loss=%s" % (i, loss.item()))

plt.ioff()
plt.show()
