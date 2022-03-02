import torch


a = torch.tensor([[1,1,1], [2,2,2]], dtype=float)
b = torch.norm(a, dim=1)

print(a)
print(a.shape)
print(b)
print(torch.sqrt(torch.tensor(3.)))