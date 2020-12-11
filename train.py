import numpy as np
import torch
from torch import optim
from torch.distributions import *
import torchvision.transforms as transforms
from torchvision import datasets
from models import VAEsimple
from data import make_dataloader 
from dataloader import load_data

bernoullisample=lambda x: np.random.binomial(1,x,size=x.shape).astype(np.float32)

class VAE(object):
    def __init__(self, input_dim, n_latent=16, lrate=1e-3, nhiddens=[256,128]):
        # model setup
        self.input_dim = input_dim
        self.n_latent = n_latent
        self.lrate = lrate
        self.cuda = torch.cuda.is_available()
        self.nhiddens=nhiddens
        self.model = VAEsimple(input_dim=self.input_dim, split_dim=7,
                         nhiddens=self.nhiddens, nlatent=self.n_latent,
                         beta=1.0, dropout=0.1, cuda=self.cuda)
        self.z0_prior = Independent(Normal(torch.zeros(self.n_latent),
                                    torch.ones(self.n_latent)), 1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lrate)

    def fit(self, data_train, data_test=None, mask_train=None, mask_test=None,
            batch_size_train=16, batch_size_test=16, nepochs=500, n_samples=3,
            wait_until_kl_inc=25):
        # model parameters
        if data_test is None:
            data_test = data_train
        if mask_train is None:
            mask_train = np.ones_like(data_train)
        if mask_test is None:
            mask_test = np.ones_like(data_test)
        # get dataset
        trainloader = make_dataloader(data_train, mask_train,
                                      batchsize=batch_size_train, shuffle=True)
        testloader = make_dataloader(data_test, mask_test,
                                     batchsize=batch_size_test, shuffle=False)
        # training
        for epoch in range(1, nepochs + 1):
            #lr = lrate * (0.8 ** (epoch // 50))
            if epoch // 1 < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1-0.96** (epoch // 1 - wait_until_kl_inc))
            loss, recloss, kld = self.model.training_func(trainloader, epoch,
                                                     self.optimizer, self.z0_prior,
                                                     kl_coef, n_samples)
    def transform(self, data, n_samples=3):
        emb = self.model.encode(torch.from_numpy(data))
        #return emb[2].mean(0).detach().numpy() # take the z and mean across n_samples
        return emb[0].detach().numpy() # take the z and mean across n_samples

    def fit_transform(self, data, nepochs=500):
        self.fit(data, nepochs=nepochs)
        return self.transform(data)
