import time, math, torch, shutil, glob
import numpy as np
import os
import random, string, os
import glob

def random_string(N=5):
    return ''.join(random.choice(string.ascii_uppercase + string.digits)
                   for _ in range(N))

def to_cuda(x):
    try:
        iterator = iter(x)
    except TypeError:
        # not iterable
        x = x.cuda()
    else:
        # iterable
        x = [x_.cuda() for x_ in x]
    return x

class OptRecorder(object):
    """collect items in optimizer"""
    def __init__(self, optimizer, n=10):
        self.opt = optimizer
        self.n = n # number of tracked parameters
        # randomly choose n parameters for each layer
        self.index = {}
        self.tracker = [] 

        for group in optimizer.param_groups:
            for p in group['params']:
                length = len(p.data.cpu().detach().numpy().ravel())
                ntrack = min(n, length)
                self.index[p] = np.random.choice(range(length), ntrack, replace=False)
                self.tracker.append({
                    "grad": [[] for _ in range(ntrack)],
                    "param": [[] for _ in range(ntrack)],
                    "alpha_ratio": [[] for _ in range(ntrack)],
                    "feature_step": [[] for _ in range(ntrack)],
                    "lr": [[] for _ in range(ntrack)],         
                })

    def record(self):
        ind = 0
        for group in self.opt.param_groups:
            for param in group['params']:
                state = self.opt.state[param]
                g = param.grad.data.cpu().detach().numpy().ravel()
                p = param.data.cpu().detach().numpy().ravel()

                if 'alpha_ratio' in state:
                    a = state['alpha_ratio'].cpu().detach().numpy().ravel()
                else:
                    a = np.ones_like(g)

                if 'feature_step' in state:
                    f = state['feature_step'].cpu().detach().numpy().ravel()
                elif 'step' in state:
                    f = np.ones_like(g) * state['step']
                else:
                    f = np.ones_like(g)

                for i, index in enumerate(self.index[param]):
                    self.tracker[ind]['grad'][i].append(g[index])
                    self.tracker[ind]['param'][i].append(p[index])                    
                    self.tracker[ind]['alpha_ratio'][i].append(a[index])
                    self.tracker[ind]['feature_step'][i].append(f[index])
                    self.tracker[ind]['lr'][i].append(group['lr'])   
                ind += 1

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum_2 = 0 # sum square
        self.var = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.sum_2 += val**2 * n
        self.var = self.sum_2 / self.count - self.avg**2

class PrintTable(object):
    '''print tabular data in a nice format'''
    def __init__(self, nstr=15, nfloat=5):
        self.nfloat = nfloat
        self.nstr = nstr

    def _format(self, x):
        if type(x) is float:
            x_tr = ("%." + str(self.nfloat) + "f") % x
        else:
            x_tr = str(x)
        return ("%" + str(self.nstr) + "s") % x_tr

    def print(self, row):
        print( "|".join([self._format(x) for x in row]) )

def smooth(sequence, step=1):
    out = np.convolve(sequence, np.ones(step), 'valid') / step
    return out

def random_split_dataset(dataset, proportions, seed=None):
    n = len(dataset)
    ns = [int(math.floor(p*n)) for p in proportions]
    ns[-1] += n - sum(ns)

    def random_split(dataset, lengths):
        if sum(lengths) != len(dataset):
            raise ValueError("Sum of input lengths does not equal\
            the length of the input dataset!")

        if seed is not None:
            np.random.seed(seed)
        indices = np.random.permutation(sum(lengths))
        return [torch.utils.data.Subset(dataset, indices[offset - length:offset])\
                for offset, length in zip(torch._utils._accumulate(lengths), lengths)]
    #return torch.utils.data.random_split(dataset, ns)
    return random_split(dataset, ns)


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

