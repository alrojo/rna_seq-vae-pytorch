# -*- coding: utf-8 -*-
import os, shutil, argparse, operator, time
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision import datasets
from tqdm import tqdm
from torch.utils.data import DataLoader

# local repos
#import data
import models

# Inspirations from https://github.com/casperkaae/LVAE
# Original paper: https://arxiv.org/abs/1602.02282
parser=argparse.ArgumentParser()
parser.add_argument("-lr", type=float,
        help="lr", default=0.002)
parser.add_argument("-outfolder", type=str,
        help="outfolder", default="dump")
parser.add_argument("-batch_size", type=int,
        help="batch_size train", default=256)
parser.add_argument("-batch_size_test", type=int,
        help="batch_size test", default=25)
parser.add_argument("-latent_sizes", type=str,
        help="latent_sizes", default="64,32")
parser.add_argument("-hidden_sizes", type=str,
        help="hidden_sizes", default="512,256")
parser.add_argument("-verbose", type=str,
        help="verbose printing", default="False")
parser.add_argument("-num_epochs", type=int,
        help="num_epochs", default=5000)
parser.add_argument("-eval_epochs", type=str,
        help="eval_epochs", default="1,10,100")
parser.add_argument("-dataset", type=str,
        help="mnistresample", default="mnistresample")
parser.add_argument("-modeltype", type=str,
        help="AE|VAE", default="VAE")
parser.add_argument("-L2", type=float,
        help="L2", default=0.0)
args=parser.parse_args()

# functions
def verbose_print(text):
    if verbose: print(text)


def plotsamples(name,outfolder,samples):
    shp=samples.shape[1:]
    nsamples=samples.shape[0]

    samples_pr_size=int(np.sqrt(nsamples))
    if len(shp)==3:
        canvas = np.zeros((h*samples_pr_size, samples_pr_size*w,shp[2]))
        cm = None
    else:
        canvas = np.zeros((h*samples_pr_size, samples_pr_size*w))
        cm = plt.gray()
    idx=0
    for i in range(samples_pr_size):
        for j in range(samples_pr_size):
            canvas[i*h:(i+1)*h, j*w:(j+1)*w] = np.clip(samples[idx],1e-6,1-1e-6)
            idx += 1
    plt.figure(figsize=(7, 7))
    plt.imshow(canvas,cmap=cm)
    plt.savefig(outfolder+'/' + name +'.png')

#setup output
if not os.path.exists(args.outfolder):
    os.makedirs(args.outfolder)

#logfile
args_dict=vars(args)
sorted_args=sorted(args_dict.items(), key=operator.itemgetter(0))
description=[]
description.append('################################')
description.append('# --Commandline Params-- #')
for name, val in sorted_args:
    description.append("# " + name + ":\t" + str(val))
description.append('################################')

scriptpath=os.path.realpath(__file__)
filename=os.path.basename(scriptpath)
#shutil.copy(scriptpath, args.outfolder + '/' + filename)
logfile=args.outfolder+'/logfile.log'
trainlogfile=args.outfolder+'/trainlogfile.log'
model_out=args.outfolder+'/model'
with open(logfile, 'w') as f:
    for l in description:
        f.write(l + '\n')

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bernoullisample=lambda x: np.random.binomial(1,x,size=x.shape).astype(np.float32)


transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(bernoullisample)
    ])
# get dataset
train_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=transform
)
val_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=transform
)
train_loader=DataLoader(
    train_data,
    batch_size=args.batch_size,
    shuffle=True
)
val_loader=DataLoader(
    train_data,
    batch_size=args.batch_size,
    shuffle=False
)
outputnonlin=torch.sigmoid

# model
if args.modeltype=='VAE':
    net=models.VAE()
else:
    raise ValueError()
net.to(device)
print(net)
optimizer=torch.optim.Adam(net.parameters(), lr=args.lr)

def train_epoch(net, dataloader, lr, epoch):
    net.train()
    running_loss={'loss': 0.0, 'KLD': 0.0, 'Reconstruction_Loss': 0.0}
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset)/dataloader.batch_size)):
        data, _ = data
        data=data.view(data.size(0), -1).to(device)
        optimizer.zero_grad()
        output=net(data)
        loss=net.loss_function(output['out'], output['x'], output['enc_mus'],
                               output['enc_log_vars'], M_N=0.001)
        running_loss['loss']+=loss['loss'].item()
        running_loss['KLD']+=loss['KLD'].item()
        running_loss['Reconstruction_Loss']+=loss['Reconstruction_Loss'].item()
        loss['loss'].backward()
        optimizer.step()
    train_loss = {key:value/len(dataloader.dataset) for key, value in running_loss.items()}
    return train_loss

def validation_epoch(net, dataloader):
    net.eval()
    running_loss={'loss': 0.0, 'KLD': 0.0, 'Reconstruction_Loss': 0.0}
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            data, _ = data
            data=data.view(data.size(0), -1).to(device)
            output=net(data)
            loss=net.loss_function(output['out'], output['x'], output['enc_mus'],
                                   output['enc_log_vars'], M_N=0.0)
            running_loss['loss']+=loss['loss'].item()
            running_loss['KLD']+=loss['KLD'].item()
            running_loss['Reconstruction_Loss']+=loss['Reconstruction_Loss'].item()
    val_loss = {key:value/len(dataloader.dataset) for key, value in running_loss.items()}
    return val_loss

val_loss=validation_epoch(net, val_loader)
line = "*Epoch=0\t" + \
       "VAL:\tCost=%0.5f\tKLD=%0.5f\tRecon=%0.5f\t"%(val_loss['loss'], val_loss['KLD'], val_loss['Reconstruction_Loss'])
print(line)
with open(trainlogfile,'a') as f:
    f.write(line + "\n")
for epoch in range(1, args.num_epochs+1):
    start=time.time()
    train_loss=train_epoch(net, train_loader, args.lr, epoch)
    t = time.time() - start
    val_loss=validation_epoch(net, val_loader)
    line = "*Epoch=%i\tTime=%0.2f\tLR=%0.5f\t" %(epoch, t, args.lr) + \
           "TRAIN:\tCost=%0.5f\tKLD=%0.5f\tRecon=%0.5f\t"%(train_loss['loss'], train_loss['KLD'], train_loss['Reconstruction_Loss']) + \
           "VAL:\tCost=%0.5f\tKLD=%0.5f\tRecon=%0.5f\t"%(val_loss['loss'], val_loss['KLD'], val_loss['Reconstruction_Loss'])
    print(line)
    with open(trainlogfile,'a') as f:
        f.write(line + "\n")
