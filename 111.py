import torch

a = torch.tensor(2, dtype=torch.float, requires_grad=True)
# y = (a*a*a).mean()
y = a*a*a

y.backward(create_graph=True)
a_grad = a.grad

# a_grad_mean = a_grad.mean()
a_grad_mean = a_grad
a.grad.data.zero_()
a_grad.backward()

a_hessian = a.grad

print('a_grad:', a_grad)
print('a_hessian', a_hessian)