import torch
from torch.autograd import Variable
from torchvision import datasets, transforms

from kindling.distributions import BernoulliNet, Normal, NormalNet
from kindling.utils import Lambda
from kindling.vae import VAE

import matplotlib.pyplot as plt

from babar.session import Session


torch.manual_seed(0)

batch_size = 128
dim_x = 784
dim_z = 20

train_loader = torch.utils.data.DataLoader(
  datasets.MNIST(
    'data/',
    train=True,
    download=True,
    transform=transforms.ToTensor(),
  ),
  batch_size=batch_size,
  shuffle=True
)
test_loader = torch.utils.data.DataLoader(
  datasets.MNIST(
    'data/',
    train=False,
    transform=transforms.ToTensor(),
  ),
  batch_size=batch_size,
  shuffle=True
)

def make_generative_net():
  return BernoulliNet(
    rate_net=torch.nn.Sequential(
      torch.nn.Linear(dim_z, 512),
      torch.nn.ReLU(),
      torch.nn.Linear(512, dim_x),
      Lambda(torch.sigmoid)
    )
  )

def make_inference_net():
  shared = torch.nn.Sequential(
    torch.nn.Linear(dim_x, 512),
    torch.nn.ReLU()
  )
  return NormalNet(
    mu_net=torch.nn.Sequential(
      shared,
      torch.nn.Linear(512, dim_z)
    ),
    sigma_net=torch.nn.Sequential(
      shared,
      torch.nn.Linear(512, dim_z),
      Lambda(lambda x: torch.exp(0.5 * x))
    )
  )

prior_z = Normal(
  Variable(torch.zeros(1, dim_z)),
  Variable(torch.ones(1, dim_z))
)

inference_net = make_inference_net()
generative_net = make_generative_net()

optimizer = torch.optim.Adam([
  {'params': set(inference_net.parameters()), 'lr': 1e-3},
  {'params': set(generative_net.parameters()), 'lr': 1e-3}
])

vae = VAE(
  inference_model=inference_net,
  generative_model=generative_net,
  prior_z=prior_z,
  optimizer=optimizer
)

def train(num_epochs, show_viz=True):
  elbo_per_iter = []

  for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(train_loader):
      info = vae.step(X=Variable(data.view(-1, dim_x)), mc_samples=1)

      elbo = info['elbo']
      z_kl = info['z_kl']
      loglik_term = info['reconstruction_log_likelihood']
      reconstr = info['reconstruction']

      if torch.max(torch.abs(reconstr.rate.data[0] - reconstr.rate.data)) < 1e-4:
        print('ya done collapsed son. try reducing your learning rate.')

      elbo_per_iter.append(elbo.data[0])
      print(f'Epoch {epoch}, {batch_idx} / {len(train_loader)}')
      print(f'  ELBO:        {elbo.data[0]}')
      print(f'  z_kl:        {z_kl.data[0]}')
      print(f'  loglik_term: {loglik_term.data[0]}')

    if show_viz:
      plt.figure()
      plt.plot(elbo_per_iter)
      plt.xlabel('iteration')
      plt.ylabel('ELBO')

      viz_reconstruction(0)
      plt.show()

def viz_reconstruction(plot_seed):
  rng_state = torch.get_rng_state()
  torch.manual_seed(plot_seed)

  # grab random sample from train_loader
  train_sample, _ = iter(train_loader).next()

  # pass through learned inference and generative networks
  q_z = inference_net(Variable(train_sample.view(-1, dim_x)))
  reconstr = generative_net(q_z.sample()).rate

  fig, ax = plt.subplots(2, 8, figsize=(12, 4))

  for i in range(8):
    ax[0, i].imshow(train_sample[i][0].numpy())
    ax[1, i].imshow(reconstr[i].data.view(28, 28).numpy())

  ax[0, 0].set_ylabel('true')
  ax[1, 0].set_ylabel('reconstructed')

  plt.tight_layout()

  torch.set_rng_state(rng_state)

if __name__ == '__main__':
  train(10)
