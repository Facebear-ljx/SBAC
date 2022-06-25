import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import d4rl
import gym
import mujoco_py

# Prepare the data
env_name = 'antmaze-medium-play-v2'
env = gym.make(env_name)
dataset = env.get_dataset()

x = dataset['observations'][0:1000000, 0]
y = dataset['observations'][0:1000000, 1]
a = np.logical_and(np.logical_and(x >= 0, x <= 0), np.logical_and(y >= 6, y <= 9))
b = np.logical_and(np.logical_and(x >= 0, x <= 0), np.logical_and(y >= 15, y <= 18))
c = np.logical_and(np.logical_and(x >= 11.5, x <= 20.5), np.logical_and(y >= 11, y <= 13))
d = np.logical_and(np.logical_and(x >= 4, x <= 13), np.logical_and(y >= 7, y <= 9))

index = np.where(np.logical_or(np.logical_or(a, b), np.logical_or(c, d)))
x = np.delete(x, index)
y = np.delete(y, index)
r = dataset['rewards']
d = dataset['terminals']
r = np.delete(r, index)
d = np.delete(d, index)
plt.scatter(x, y)
plt.show()
# t-SNE
X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(x)

# Data Visualization
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # Normalize
plt.figure(figsize=(8, 8))
plt.scatter(X_norm[:, 0], X_norm[:, 1])
# for i in range(X_norm.shape[0]):
#     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
#              fontdict={'weight': 'bold', 'size': 9})

plt.xticks([])
plt.yticks([])
plt.show()
