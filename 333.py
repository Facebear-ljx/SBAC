import torch
from Network.Actor_Critic_net import W

w_net = W(1, 2, 'cpu').to('cpu')
w_optim = torch.optim.SGD(w_net.parameters(), lr=1e-5)
a = torch.tensor([2], dtype=torch.float, requires_grad=False)

w = w_net(a)
y = w * w

w_optim.zero_grad()
y.backward(create_graph=True)
# w_optim.step()
a_grad = w_net.parameters().grad

# a_grad_mean = a_grad.mean()
a_grad_mean = a_grad
a.grad.data.zero_()
a_grad.backward()

a_hessian = a.grad

print('a_grad:', a_grad)
print('a_hessian', a_hessian)