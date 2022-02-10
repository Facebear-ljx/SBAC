import torch
from Network.Actor_Critic_net import W, Actor_deterministic
import wandb

wandb.init(project="222", entity="facebear")
MIN = -100.
MAX = 100.


def dual_descent(grad_energy, lmbda):
    lmbda_loss = -grad_energy * lmbda
    lmbda_loss = lmbda_loss.mean()
    lmbda_optim.zero_grad()
    lmbda_loss.backward(retain_graph=True)
    lmbda_optim.step()
    return lmbda


s = torch.tensor([2, 2, 2], dtype=torch.float, requires_grad=False)

EBM = Actor_deterministic(3, 1, 3, 'cpu')
actor = Actor_deterministic(3, 3, 3, 'cpu')
actor_optim = torch.optim.Adam(actor.parameters(), lr=3e-5)

lmbda = torch.tensor(torch.ones(1, 3), dtype=torch.float, requires_grad=True)
lmbda_optim = torch.optim.Adam([lmbda], lr=1e-4)

for i in range(50000):
    a = actor(s)  # optimal is [0, 0, 0]
    # energy = EBM(a)
    energy = (torch.min(a * a, (a-0.5)*(a-0.5))).mean()

    de_da = torch.autograd.grad(energy, a, create_graph=True)

    de_da = de_da[0]
    # de_da = de_da.mean()
    lmbda = dual_descent(de_da, lmbda)

    actor_loss = de_da * lmbda
    actor_loss = actor_loss.mean()
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

    wandb.log({"de_da": de_da.mean().item(),
               "energy": energy.item(),
               "lmbda": lmbda.mean().item(),
               "actor_loss": actor_loss,
               "a_mean": a.mean().item()})
    # a_hessian = a.grad
    # if i % 100 == 0:
    #     print('\n---------------')
    #     print('a_grad:', actor_loss)
    #     print('energy:', energy)
    #     print('lmbda:', lmbda)
    #     print('a', a)
