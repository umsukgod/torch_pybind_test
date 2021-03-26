import torch

a = torch.rand(2, 3)
print(a)

torch.load("../current_0.pt",map_location=lambda storage, loc: storage)
