# coding: utf-8

# In[224]:

import argparse
parser = argparse.ArgumentParser(description="STN")
parser.add_argument('-c', type=float,
                    help='cosine similarity', default=1)
parser.add_argument('-m', type=str,
                    help='model save directory', default='models')
args = parser.parse_args()

import numpy as np
import matplotlib.pyplot as plt

def generate_tasks(p=0.5, d=100, c=1, n=300, alphas=[], betas=[]):
    '''
    d: dimensionality
    c: scaling factor
    p: correlation
    n: number of data points
    '''
    assert d >= 2, "at least 2 dimension"    
    u1, u2 = np.zeros(d), np.zeros(d)
    u1[0] = 1
    u2[1] = 1
    w1 = c * u1
    w2 = c * (p * u1 + np.sqrt(1-p**2) * u2)
    
    X = []
    Y1 = []
    Y2 = []
    for _ in range(n):
        x = np.random.normal(0, 1, d)
        y1 = w1.dot(x) + np.random.normal(0,0.01)
        y2 = w2.dot(x) + np.random.normal(0,0.01)
        for a, b in zip(alphas, betas):
            y1 += np.sin(a * w1.dot(x) + b)
            y2 += np.sin(a * w2.dot(x) + b)
            
        X.append(x)
        Y1.append(y1)
        Y2.append(y2)
    return np.vstack(X), np.array(Y1), np.array(Y2)

X, Y1, Y2 = generate_tasks(p=0,alphas=[1,2],betas=[3,4])

# In[179]:


import torch
import torch.nn as nn
from lib.model import SharedBottom, Independent, MMOE
torch.set_num_threads(1)

    
# In[180]:


from torch.utils.data import DataLoader  
from torch.utils.data import Dataset

class CosDataset(Dataset):
    
    def __init__(self, p=0.5, d=100, c=1, n=300, alphas=[], betas=[]):
        '''
        d: dimensionality
        c: scaling factor
        p: correlation
        n: number of data points
        '''
        self.X, self.Y1, self.Y2 = generate_tasks(p=p,d=d,c=c,n=n,
                                                  alphas=alphas,betas=betas)
        self.X = torch.from_numpy(self.X).float()
        
    def __len__(self):
        return len(self.Y1)
    
    def __getitem__(self, idx):
        return self.X[idx], (self.Y1[idx], self.Y2[idx])
    
    
def MTL_loss():
    
    def c(yhat, y):
        # regression loss on 2 tasks
        y1, y2 = y
        yhat1, yhat2 = yhat
        c_ = nn.MSELoss()
        return c_(yhat1.view(-1), y1.float().view(-1)) + c_(yhat2.view(-1), y2.float().view(-1))
        
    return c

# In[217]:


from torch.utils.data import DataLoader
cos_sim = args.c
n = 10000
alphas = [1, 2]
betas = [3, 4]
train_data = DataLoader(CosDataset(p=cos_sim, n=n, alphas=alphas, betas=betas),
                        batch_size=min(1000, n), num_workers=0)
val_data = DataLoader(CosDataset(p=cos_sim, n=n, alphas=alphas, betas=betas),
                      batch_size=min(1000, n), num_workers=0)


# In[218]:

from lib.train import TrainFeedForward
import os
modeldir = '{}/c={}'.format(args.m, args.c)
os.system('mkdir -p {}'.format(modeldir))

nets = [Independent(0), Independent(1), Independent(2),
        SharedBottom(0), SharedBottom(1), SharedBottom(2), MMOE()]
trainers = []
for net in nets:
    net_name = net.name()
    print(net_name)
    t = TrainFeedForward(net, train_data, val_data=val_data, criterion=MTL_loss(),
                         n_iters=1000, n_save=30,
                         save_filename='{}/{}.pth.tar'.format(modeldir,net_name))
    trainers.append(t)
    t.train()






