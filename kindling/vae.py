import torch
from torch.autograd import Variable

from .utils import KL_Normals


class VAE(object):
  """An implementation of a variational autoenoder. Note that this particular
  implementation requires that the prior over z and the approximate posterior
  q(z | x) must both be diagonal normal distributions."""

  def __init__(
      self,
      inference_model,
      generative_model,
      prior_z
  ):
    self.inference_model = inference_model
    self.generative_model = generative_model
    self.prior_z = prior_z

  def evaluate_elbo(self, X, mc_samples):
    batch_size = X.size(0)

    # [batch_size, dim_z]
    q_z = self.inference_model(X)

    # KL divergence is additive across independent joint distributions, so this
    # works appropriately. This is where the normal constraints come from.
    z_kl = KL_Normals(q_z, self.prior_z.expand_as(q_z)) / batch_size

    # [batch_size * mc_samples, dim_z]
    z_sample = torch.cat([q_z.sample() for _ in range(mc_samples)], dim=0)
    Xrep = Variable(X.data.repeat(mc_samples, 1))
    reconstr = self.generative_model(z_sample)
    loglik_term = (
      reconstr.logprob(Xrep)
      / mc_samples
      / batch_size
    )

    elbo = -z_kl + loglik_term

    return {
      'elbo': elbo,
      'z_kl': z_kl,
      'reconstruction_log_likelihood': loglik_term,
      'z_sample': z_sample,
      'reconstruction': reconstr
    }
