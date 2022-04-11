import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import d4rl
import gym
import mujoco_py

# Prepare the data
env_name = 'antmaze-large-play-v2'
env = gym.make(env_name)
dataset = env.get_dataset()

x = dataset['observations'][0:100000, 0]
y = dataset['observations'][0:100000, 1]
index = np.where(np.logical_and(np.logical_and(x>=10, x<=17), np.logical_and(y>=10, y<=15)))
x = np.delete(x, index)
y = np.delete(y, index)
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
