import torch


def KL_Normals(d0, d1):
  return torch.sum(KL_Normals_independent(d0, d1))

# See https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
def KL_Normals_independent(d0, d1):
  assert d0.mu.size() == d1.mu.size()
  sigma0_sqr = torch.pow(d0.sigma, 2)
  sigma1_sqr = torch.pow(d1.sigma, 2)
  return (
    -0.5
    + (sigma0_sqr + torch.pow(d0.mu - d1.mu, 2)) / (2 * sigma1_sqr)
    + torch.log(d1.sigma)
    - torch.log(d0.sigma)
  )

class Lambda(torch.nn.Module):
  def __init__(self, func, extra_args=(), extra_kwargs={}):
    super(Lambda, self).__init__()
    self.func = func
    self.extra_args = extra_args
    self.extra_kwargs = extra_kwargs

  def forward(self, x):
    return self.func(x, *self.extra_args, **self.extra_kwargs)

class MetaOptimizer(object):
  def __init__(self, optimizers):
    self.optimizers = optimizers

  def zero_grad(self):
    for opt in self.optimizers:
      opt.zero_grad()

  def step(self):
    for opt in self.optimizers:
      opt.step()
