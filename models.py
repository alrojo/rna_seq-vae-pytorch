import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.a=F.leaky_relu
        # encoder
        self.l_enc_h_1 = nn.Linear(784, 512)
        self.l_enc_h_2 = nn.Linear(512, 512)
        self.l_enc_mu = nn.Linear(512, 64)
        self.l_enc_var = nn.Linear(512, 64)

        # decoder
        self.l_dec_h_1 = nn.Linear(64, 512)
        self.l_dec_h_2 = nn.Linear(512, 512)

        # output
        self.out = nn.Linear(512, 784)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x):
        x=self.a(self.l_enc_h_1(x)) 
        x=self.a(self.l_enc_h_2(x)) 
        mu=self.l_enc_mu(x)
        log_var=self.l_enc_var(x)
        z=self.reparameterize(mu, log_var)

        return [z], [mu], [log_var]

    def decode(self, x):
        x=self.a(self.l_dec_h_1(x)) 
        x=self.a(self.l_dec_h_2(x)) 

        # output
        out=self.out(x)
        return out, [], [], []

    def forward(self, x):
        enc_zs, enc_mus, enc_log_vars=self.encode(x) 
        out, dec_zs, dec_mus, dec_log_vars=self.decode(enc_zs[-1]) 
        formatted = {'out': out, 'x': x, 'enc_mus': enc_mus,
                     'enc_log_vars': enc_log_vars, 'dec_mus': dec_mus,
                     'dec_log_vars': dec_log_vars, 'enc_zs': enc_zs,
                     'dec_zs': dec_zs}

        return formatted

    def loss_function(self, *args, **kwargs):
        recons=args[0]
        input=args[1]
        mu=args[2][0]
        log_var=args[3][0]

        kld_weight=kwargs['M_N']
        recons_loss=F.mse_loss(recons, input)

        kld_loss=torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        kld_re=kld_weight*kld_loss
        #print(kld_re)

        loss=recons_loss+kld_re
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD':-kld_loss}

    def sample(self, num_samples, current_device):
        z=torch.randn(num_samples. self.latent_dim)
        z=z.to(current_device)
        samples=self.decode(z)
        return samples


class hVAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        self.a=F.leaky_relu
        # encoder
        self.l_enc_h_11 = nn.Linear(784, 512)
        self.l_enc_h_12 = nn.Linear(512, 512)
        self.l_enc_mu_1 = nn.Linear(512, 64)
        self.l_enc_var_1 = nn.Linear(512, 64)

        self.l_enc_h_21 = nn.Linear(64, 256)
        self.l_enc_h_22 = nn.Linear(256, 256)
        self.l_enc_mu_2 = nn.Linear(256, 32)
        self.l_enc_var_2 = nn.Linear(256, 32)
        
        # decoder
        self.l_dec_h_21 = nn.Linear(32, 256)
        self.l_dec_h_22 = nn.Linear(256, 256)

        self.l_dec_mu_1 = nn.Linear(256, 64)
        self.l_dec_var_1 = nn.Linear(256, 64)
        self.l_dec_h_11 = nn.Linear(64, 512)
        self.l_dec_h_12 = nn.Linear(512, 512)

        # output
        self.out = nn.Linear(512, 784)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x):
        # layer enc 1
        x=self.a(self.l_enc_h_11(x)) 
        x=self.a(self.l_enc_h_12(x)) 
        mu1=self.l_enc_mu_1(x)
        logvar1=self.l_enc_var_1(x)
        z1=self.reparameterize(mu1, logvar1)

        # layer enc 2
        x=self.a(self.l_enc_h_21(z1)) 
        x=self.a(self.l_enc_h_22(x)) 
        mu2=self.l_enc_mu_2(x)
        logvar2=self.l_enc_var_2(x)
        z2=self.reparameterize(mu2, logvar2)
        return [z1, z2], [mu1, mu2], [logvar1, logvar2]

    def decode(self, x):
        # layer dec 2
        x=self.a(self.l_dec_h_21(x)) 
        x=self.a(self.l_dec_h_22(x)) 

        # layer dec 1
        mu1=self.l_dec_mu_1(x)
        logvar1=self.l_dec_var_1(x)
        z1=self.reparameterize(mu1, logvar1)
        x=self.a(self.l_dec_h_11(z1)) 
        x=self.a(self.l_dec_h_12(x)) 

        # output
        out=self.out(x)
        return out, [z1], [mu1], [logvar1]

    def forward(self, x):
        enc_zs, enc_mus, enc_log_vars=self.encode(x) 
        out, dec_zs, dec_mus, dec_log_vars=self.decode(enc_zs[-1]) 

        return [out, enc_mus, enc_log_vars, dec_mus, dec_log_vars]

if __name__ == "__main__":
    net=VAE()
    import numpy as np
    sample=np.random.normal(0,1,(10,784))
    sample=np.float32(sample)
    out=net(torch.from_numpy(sample))
    for ou in out:
        if isinstance(ou, list):
            for o in ou:
                print(o.shape)
        else:
            print(ou.shape)
