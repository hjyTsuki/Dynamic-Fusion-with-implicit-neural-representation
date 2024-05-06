import torch

a = torch.Tensor([1.0])
a.requires_grad = True
b = torch.Tensor([2.0])
b.requires_grad = True
c = a + b
c.backward()
print(a)