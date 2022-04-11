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

n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)

x_circles = torch.unsqueeze(torch.Tensor(noisy_circles[0][:, 0]), dim=1)
y_circles = torch.unsqueeze(torch.Tensor(noisy_circles[0][:, 1]), dim=1)

x_moons = torch.unsqueeze(torch.Tensor(noisy_moons[0][:, 0]), dim=1)
y_moons = torch.unsqueeze(torch.Tensor(noisy_moons[0][:, 1]), dim=1)

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x * x  # + 0.2 * torch.rand(x.size())

class AE_convex_transform(nn.Module):
    def __init__(self, dim_action, dim_hidden, dim_latent):
        super(AE_convex_transform, self).__init__()
        self.en1 = nn.Linear(dim_action, dim_hidden)
        self.en2 = nn.Linear(dim_hidden, dim_hidden)
        self.en3 = nn.Linear(dim_hidden, dim_latent)

        self.de1 = nn.Linear(dim_latent, dim_hidden)
        self.de2 = nn.Linear(dim_hidden, dim_hidden)
        self.de3 = nn.Linear(dim_hidden, dim_action)

    def encoder(self, action):
        x = F.relu(self.en1(action))
        x = F.relu(self.en2(x))
        latent = self.en3(x)
        return latent

    def decoder(self, latent):
        x = F.relu(self.de1(latent))
        x = F.relu(self.de2(x))
        x = self.de3(x)
        return x

    def forward(self, action):
        latent = self.encoder(action)
        action_recon = self.decoder(latent)
        return action_recon, latent

class encoder(nn.Module):
    def __init__(self, dim_action, dim_hidden, dim_latent):
        super(encoder, self).__init__()
        self.en1 = nn.Linear(dim_action, dim_hidden)
        self.en2 = nn.Linear(dim_hidden, dim_hidden)
        self.en3 = nn.Linear(dim_hidden, dim_latent)

    def forward(self, action):
        x = F.relu(self.en1(action))
        x = F.relu(self.en2(x))
        latent = self.en3(x)
        return latent

model = AE_convex_transform(2, 128, 2).to('cpu')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model2 = encoder(2, 64, 2).to('cpu')
optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)

# plot
x1 = (torch.rand(size=[500, 1])-0.5)* 6
y1 = (torch.rand(size=[500, 1])-0.5)* 6
# x1, y1 = torch.meshgrid(x1, y1)
# ax = Axes2D(fig)

input_train = torch.cat([x_circles, y_circles], dim=1)
input_test = torch.cat([x1, y1], dim=1)
# train and plot
pbar = tqdm(range(10000))
# for i in pbar:
#     recon, latent = model(input_train)
#     recon_noise, latent_noise = model(input_test)
#
#     recon_loss = nn.MSELoss()(input_train, recon)
#     recon_loss_noise = nn.MSELoss()(input_test, recon_noise) * 0.1
#     dis_loss = 0.005 * (-1 * torch.norm(latent) + torch.norm(latent_noise))
#
#     loss = recon_loss + recon_loss_noise + dis_loss
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

for i in pbar:
    latent = model2(input_train)
    latent_2 = model2(input_test)
    dis_loss = (-torch.norm(latent) * 0.0 + torch.norm(latent_2)) * 0.1
    recon_loss = nn.MSELoss()(latent, input_train) + nn.MSELoss()(latent_2, input_test)
    loss = dis_loss + recon_loss
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()

    if i % 500 == 0:
        with torch.no_grad():
            plt.cla()

            plt.subplot(1, 3, 1)
            x1 = input_train[:, 0].numpy()
            y1 = input_train[:, 1].numpy()

            x2 = input_test[:, 0].numpy()
            y2 = input_test[:, 1].numpy()
            plt.scatter(x1, y1, c='blue')
            plt.scatter(x2, y2, c='red')

            latent = model2(input_train)
            x1 = latent[:, 0].numpy()
            y1 = latent[:, 1].numpy()

            latent_2 = model2(input_test)
            x2 = latent_2[:, 0].numpy()
            y2 = latent_2[:, 1].numpy()

            plt.subplot(1, 3, 2)
            plt.scatter(x1, y1, c='blue')
            plt.scatter(x2, y2, c='red')

            plt.subplot(1, 3, 3)
            # x3 = recon[:, 0].numpy()
            # y3 = recon[:, 1].numpy()
            # x4 = recon2[:, 0].numpy()
            # y4 = recon2[:, 1].numpy()
            # plt.scatter(x3, y3, c='blue')
            # plt.scatter(x4, y4, c='red')

            plt.pause(0.01)

    pbar.set_description("Processing%s, loss=%s" % (i, loss.item()))


# ax.set_title('energy surface')
# ax.set_xlabel('input')
# ax.set_ylabel('output')
# ax.set_zlabel('energy')
plt.ioff()
plt.show()
