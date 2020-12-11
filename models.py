import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import *

class VAEsimple(nn.Module):
    def __init__(self, input_dim=None, split_dim=None, nhiddens=[200,200], nlatent=20,
                 beta=1.0, dropout=0.2, n_samples=3, cuda=False):

        if nlatent < 1:
            raise ValueError('Minimum 1 latent neuron, not {}'.format(latent))

        if not (0 <= dropout < 1):
            raise ValueError('dropout must be 0 <= dropout < 1')
        self.split_dim = split_dim
        self.input_size = input_dim

        super(VAEsimple, self).__init__()

        # Initialize simple attributes
        self.usecuda = cuda
        self.beta = beta
        self.nhiddens = nhiddens
        self.nlatent = nlatent
        self.dropout = dropout
        self.n_samples = n_samples

        self.device = torch.device("cuda" if self.usecuda == True else "cpu")

        # Activation functions
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.dropoutinput = nn.Dropout(p=self.dropout*0.5)

        # Initialize lists for holding hidden layers
        self.encoderlayers = nn.ModuleList()
        self.encodernorms = nn.ModuleList()
        self.encoderdrops = nn.ModuleList()
        self.decoderlayers = nn.ModuleList()
        self.decodernorms = nn.ModuleList()
        self.decoderdrops = nn.ModuleList()

        ### Layers
        # Hidden layers
        for nin, nout in zip([self.input_size] + self.nhiddens, self.nhiddens):
            self.encoderlayers.append(nn.Linear(nin, nout))
            self.encoderdrops.append(nn.Dropout(p=self.dropout))
            #self.encodernorms.append(nn.BatchNorm1d(nout))

        # Latent layers
        self.mu = nn.Linear(self.nhiddens[-1], self.nlatent) # mu layer
        self.var = nn.Linear(self.nhiddens[-1], self.nlatent) # logvariance layer

        # Decoding layers
        for nin, nout in zip([self.nlatent] + self.nhiddens[::-1], self.nhiddens[::-1]):
            self.decoderlayers.append(nn.Linear(nin, nout))
            #self.decodernorms.append(nn.BatchNorm1d(nout))
            self.decoderdrops.append(nn.Dropout(p=self.dropout))

        # Reconstruction - output layers
        self.out = nn.Linear(self.nhiddens[0], self.input_size) #to output

    def encode(self, tensor):
        tensors = list()
        # hidden layers
        for encoderlayer, encoderdrop in zip(self.encoderlayers, self.encoderdrops):
            tensor = encoderdrop(self.relu(encoderlayer(tensor)))
            tensors.append(tensor)

        return self.mu(tensor), self.softplus(self.var(tensor))

    def reparametize(self, mu, std, n_samples):
        q_z = Independent(Normal(mu, std), 1)

        return q_z, q_z.rsample(torch.Size([n_samples]))

    def decode(self, tensor, n_samples):
        tensors = list()

        for decoderlayer, decoderdrop in zip(self.decoderlayers, self.decoderdrops):
            tensor = decoderdrop(self.relu(decoderlayer(tensor)))
            tensors.append(tensor)

        reconstruction = self.out(tensor)

        return reconstruction

    def forward(self, tensor, n_samples):
        mu, std = self.encode(tensor)
        q_z, z = self.reparametize(mu, std, n_samples)
        out = self.decode(z, n_samples)

        return out, q_z, z

    # Reconstruction loss + KL Divergence losses summed over all elements and batch
    def loss_function(self, input_data, output, mask, q_z, p_z, kld_w):
        # Repeat data by n_samples
        n_samples = output.shape[0]
        input_data = input_data.repeat(n_samples,1,1)
        mask = mask.repeat(n_samples,1,1)

        mse_loss = nn.MSELoss(reduction='none')
        rec_loss = (mse_loss(output, input_data)*mask).sum()/mask.sum()

        # KLD calculation and weight
        KLD = (kl_divergence(q_z, p_z)/self.nlatent).mean()
        KLD_weight = self.beta * kld_w

        # Final loss
        loss = rec_loss + KLD * KLD_weight

        return loss, rec_loss, KLD

    def training_func(self, train_loader, epoch, optimizer, p_z, kld_w, n_samples):

        self.train()
        train_loss = 0
        log_interval = 50

        epoch_loss = 0
        epoch_kldloss = 0
        epoch_recloss = 0

        for batch_idx, (data, mask) in enumerate(train_loader):
            data = data.to(self.device).float()
            mask = mask.to(self.device).float()

            optimizer.zero_grad()

            out, q_z, _ = self(data, n_samples)

            loss, rec_loss, kld = self.loss_function(data, out, mask, q_z, p_z, kld_w)
            loss.backward()

            epoch_loss += loss.data.item()
            epoch_kldloss += kld.data.item()
            epoch_recloss += rec_loss.data.item()

            optimizer.step()

        print('\tEpoch: {}\tLoss: {:.6f}\tRec: {:.6f}\tKLD: {:.4f}\tBatchsize: {}'.format(
            epoch,
            epoch_loss / len(train_loader),
            epoch_recloss / len(train_loader),
            epoch_kldloss / len(train_loader),
            train_loader.batch_size,
            ))
        return epoch_loss / len(train_loader), epoch_recloss / len(train_loader), epoch_kldloss / len(train_loader)
