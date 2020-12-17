import torch
from torch import tensor
from torch.distributions import Normal, MultivariateNormal
from math import sqrt, exp, log

PI = 3.141592653
TAU = 6.28318530717958

mu = tensor([0.])
std = tensor([2.])

# mu = torch.rand(10, 2)
# std = torch.rand(10, 2)
#
normal = Normal(mu, std)
m_normal = MultivariateNormal(mu, torch.diag_embed(std**2))

# samples = normal.sample()

def normal_logprob(x, mu, std):
    return log(1./(std*sqrt(TAU)) * exp(-0.5 * ((x - mu) / std)**2))

# print(f"Normal: {normal.log_prob(samples).sum(1)}")
# print(f"M-Normal: {m_normal.log_prob(samples)}")

print(f"Torch normal: {normal.log_prob(tensor(0.)).item()}")
print(f"Torch mnormal: {m_normal.log_prob(tensor(0.)).item()}")
print(f"Real: {normal_logprob(0, 0, 2)}")