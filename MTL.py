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

# In[174]:


print('number of parameters comparison')
n_neuron = 50
print('shared bottom net:', 100 * 113 + 113 * 8 + 113 * 8 + 8 + 8)
print('deep shared bottom net:', 100 * n_neuron + n_neuron*n_neuron*3 + n_neuron*8*2 + 8*2)
print('deep independent net:', (100 * 32 + 32*32*3 + 32*8 + 8)*2)
print('MMOE:', 100 * 16 * 8 + 16 * 8 * 2 + 8 + 8 + 100 * 8 * 2)


# In[179]:


import torch
import torch.nn as nn

class SharedBottom(nn.Module):

    def __init__(self, l=4):
        '''
        l: number of layers
        '''
        super(SharedBottom, self).__init__()
        self.l = l
        if l == 1:
            a = int(np.round((14672 - 16) / 116))
        else:
            a = int(np.round((-116 + np.sqrt(116**2 + 4*(l-1)*(14672-16))) / (2 * (l-1))))

        print('per layer neuron: {}'.format(a))
        base = [nn.Linear(100, a), nn.ReLU(inplace=True)]
        for _ in range(l-1):
            base.extend([nn.Linear(a, a), nn.ReLU(inplace=True)])
        self.bottom = nn.Sequential(*base)    

        self.top1 = nn.Sequential(
            nn.Linear(a, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1)
        )
        self.top2 = nn.Sequential(
            nn.Linear(a, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1)
        )
        
    def forward(self, x):
        shared = self.bottom(x)
        return self.top1(shared), self.top2(shared)
    
    def name(self):
        return 'SharedBottom(l={})'.format(self.l)

class Independent(nn.Module):

    def __init__(self, l=4):
        '''
        l: number of layers
        '''
        super(Independent, self).__init__()
        self.l = l
        if l == 1:
            a = int(np.round((14672/2-8) / 108))
        else:
            a = int(np.round((-108 + np.sqrt(108**2 + 4*(l-1)*(14672/2-8))) / (2 * (l-1))))

        print('per layer neuron: {}'.format(a))
        base1 = [nn.Linear(100, a), nn.ReLU(inplace=True)]
        base2 = [nn.Linear(100, a), nn.ReLU(inplace=True)]
        for _ in range(l-1):
            base1.extend([nn.Linear(a, a), nn.ReLU(inplace=True)])
            base2.extend([nn.Linear(a, a), nn.ReLU(inplace=True)])
        base1.extend([
            nn.Linear(a, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1)
        ])
        base2.extend([
            nn.Linear(a, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1)            
        ])
        self.top1 = nn.Sequential(*base1)
        self.top2 = nn.Sequential(*base2)

    def name(self):
        return 'Independent(l={})'.format(self.l)
        
    def forward(self, x):
        return self.top1(x), self.top2(x)

class MMOE(nn.Module):

    def __init__(self):
        super(MMOE, self).__init__()

        n_experts = 8
        self.experts = nn.ModuleList()
        for _ in range(n_experts):
            self.experts.append(
                nn.Sequential(
                    nn.Linear(100, 16),
                    nn.ReLU(inplace=True)
                )
            )

        self.gates = nn.ModuleList()
        for _ in range(2): # 2 tasks
            self.gates.append(
                 nn.Sequential(
                    nn.Linear(100, n_experts),
                    nn.Softmax(dim=1)
                )            
            )
        
        
        self.top1 = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1)
        )
        self.top2 = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1)
        )
        
    def use_gate(self, outputs, gate_outputs):
        '''
        gate_outputs: (n, n_experts)
        '''
        res = 0
        for i in range(len(self.experts)):
            res += gate_outputs[:, i:(i+1)] * outputs[i]
        return res
        
    def forward(self, x):
        outputs = [self.experts[i](x) for i in range(len(self.experts))]
        gate_outputs = [self.gates[i](x) for i in range(2)]
        task1x = self.use_gate(outputs, gate_outputs[0])
        task2x = self.use_gate(outputs, gate_outputs[1])
        return self.top1(task1x), self.top2(task2x)
    
    def name(self):
        return "MMOE()"
    
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
train_data = DataLoader(CosDataset(p=cos_sim, n=n, alphas=alphas, betas=betas), batch_size=min(1000, n))
val_data = DataLoader(CosDataset(p=cos_sim, n=n, alphas=alphas, betas=betas), batch_size=min(1000, n))


# In[218]:

from lib.train import TrainFeedForward
import os
modeldir = '{}/c={}'.format(args.m, args.c)
os.system('mkdir -p {}'.format(modeldir))
nets = [Independent(1), Independent(2), SharedBottom(1), SharedBottom(2), MMOE()]
trainers = []
for net in nets:
    net_name = net.name()
    print(net_name)
    t = TrainFeedForward(net, train_data, val_data=val_data, criterion=MTL_loss(), n_iters=1000,
                         save_filename='{}/{}.pth.tar'.format(modeldir,net_name), n_save=30)
    trainers.append(t)
    t.train()






