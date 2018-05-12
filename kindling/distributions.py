from abc import ABC, abstractmethod
import itertools
import math

import torch
from torch.autograd import Variable
from torch.nn import functional as F

# from .utils import split


LOG2PI = torch.log(torch.FloatTensor([2 * math.pi]))[0]

# class NormalLogStdDev(object):
#   def __init__(self, mean, log_stddev):
#     assert mean.size() == log_stddev.size()
#     self.mean = mean
#     self.log_stddev = log_stddev

#   def sample(self):
#     return self.mean + torch.exp(self.log_stddev) * Variable(torch.randn(self.mean.size()))

#   def logprob(self, x):
#     return -0.5 * (
#       self.mean.numel() * LOG2PI
#       + 2 * torch.sum(self.log_stddev)
#       + torch.sum((x - self.mean) * torch.exp(-2 * self.log_stddev) * (x - self.mean))
#     )

class Distribution(ABC):
  @abstractmethod
  def size(self, *args, **kwargs):
    pass

  @abstractmethod
  def sample(self):
    pass

  def logprob(self, x):
    return torch.sum(self.logprob_independent(x))

  @abstractmethod
  def logprob_independent(self, x):
    pass

class Normal(Distribution):
  def __init__(self, mu, sigma):
    super().__init__()
    assert mu.size() == sigma.size()
    self.mu = mu
    self.sigma = sigma

  def size(self, *args, **kwargs):
    return self.mu.size(*args, **kwargs)

  def sample(self):
    eps = torch.randn(self.mu.size()).type_as(self.mu.data)
    return self.mu + self.sigma * Variable(eps)

  def logprob_independent(self, x):
    return (
      -0.5 * LOG2PI
      - torch.log(self.sigma)
      -0.5 * torch.pow((x - self.mu) / self.sigma, 2)
    )

  def expand(self, *args, **kwargs):
    return Normal(
      self.mu.expand(*args, **kwargs),
      self.sigma.expand(*args, **kwargs)
    )

  def expand_as(self, *args, **kwargs):
    return Normal(
      self.mu.expand_as(*args, **kwargs),
      self.sigma.expand_as(*args, **kwargs)
    )

class Normal_MeanPrecision(Distribution):
  def __init__(self, mu, precision):
    super().__init__()
    assert mu.size() == precision.size()
    self.mu = mu
    self.precision = precision

  def size(self, *args, **kwargs):
    return self.mu.size(*args, **kwargs)

  def sample(self):
    eps = torch.randn(self.mu.size()).type_as(self.mu.data)
    return self.mu + self.precision.pow(-0.5) * Variable(eps)

  def logprob_independent(self, x):
    return (
      -0.5 * LOG2PI
      +0.5 * torch.log(self.precision)
      -0.5 * torch.pow((x - self.mu) * torch.sqrt(self.precision), 2)
    )

class Bernoulli(Distribution):
  def __init__(self, rate):
    super().__init__()
    self.rate = rate

  def size(self, *args, **kwargs):
    return self.rate.size(*args, **kwargs)

  def sample(self):
    return Variable(torch.rand(self.rate.size())) < self.rate

  def logprob(self, x):
    return -F.binary_cross_entropy(self.rate, x, size_average=False)

  def logprob_independent(self, x):
    return x * torch.log(self.rate) + (1 - x) * torch.log(1 - self.rate)

class Gamma(Distribution):
  def __init__(self, shape, rate):
    assert shape.size() == rate.size()
    super().__init__()
    self.shape = shape
    self.rate = rate

  def size(self, *args, **kwargs):
    return self.shape.size(*args, **kwargs)

  def logprob_independent(self, x):
    return (
      self.shape * torch.log(self.rate)
      - torch.lgamma(self.shape)
      + (self.shape - 1) * torch.log(x)
      - self.rate * x
    )

class NormalNet(object):
  def __init__(self, mu_net, sigma_net):
    self.mu_net = mu_net
    self.sigma_net = sigma_net

  def __call__(self, x):
    return Normal(self.mu_net(x), self.sigma_net(x))

  def parameters(self):
    return itertools.chain(
      self.mu_net.parameters(),
      self.sigma_net.parameters()
    )

#   def cuda(self):
#     self.mu_net.cuda()
#     self.sigma_net.cuda()

class Normal_MeanPrecisionNet(object):
  def __init__(self, mu_net, precision_net):
    self.mu_net = mu_net
    self.precision_net = precision_net

  def __call__(self, x):
    return Normal_MeanPrecision(self.mu_net(x), self.precision_net(x))

  def parameters(self):
    return itertools.chain(
      self.mu_net.parameters(),
      self.precision_net.parameters()
    )

class BernoulliNet(object):
  def __init__(self, rate_net):
    self.rate_net = rate_net

  def __call__(self, x):
    return Bernoulli(self.rate_net(x))

  def parameters(self):
    return self.rate_net.parameters()

# class _DistributionCat(object):
#   def __init__(self, distributions, dim=-1):
#     self.distributions = distributions
#     self.dim = dim

#   def sample(self):
#     return torch.cat([d.sample() for d in self.distributions], dim=self.dim)

#   def logprob(self, x):
#     xs = split(x, [d.size(self.dim) for d in self.distributions], dim=self.dim)
#     return sum(d.logprob(xx) for d, xx in zip(self.distributions, xs))

#   def entropy(self):
#     return sum(d.entropy() for d in self.distributions)

# def DistributionCat(distributions, dim=-1):
#   if all(isinstance(d, Normal) for d in distributions):
#     return Normal(
#       mu=torch.cat([d.mu for d in distributions], dim=dim),
#       sigma=torch.cat([d.sigma for d in distributions], dim=dim)
#     )
#   else:
#     return _DistributionCat(distributions, dim=-1)
