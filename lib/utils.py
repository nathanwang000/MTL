import time, math, torch, shutil, glob
import numpy as np
import os
import random, string, os
import glob, copy

def crossed_zero(old, new):
    return (old * new <= 0).float()

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

class CrossZeroTracker(object):
    def __init__(self, optimizer):
        self.opt = optimizer
        self.beta1 = 0.9
        self.n_grad_flip = 0 # number of momentum flip
        self.momentum = {}
        for group in self.opt.param_groups:
            for p in group['params']:
                self.momentum[p] = torch.zeros_like(p)
                
    def record(self):
        for group in self.opt.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                
                exp_avg = self.momentum[p]
                self.n_grad_flip += torch.sum(crossed_zero(exp_avg,
                                                           exp_avg*self.beta1\
                                                           + (1-self.beta1)*grad)).item()
                exp_avg.mul_(self.beta1).add_(1 - self.beta1, grad)
        
class OptRecorder(object):
    """collect items in optimizer"""
    def __init__(self, optimizer, n=10, model=None):
        if model is not None:
            self.w0 = copy.deepcopy(model.state_dict())
            
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

        # last item of tracker keeps track of global properties
        if model is not None:
            self.tracker.append({"l2(w-w0)": [], "l2(w)": []})

    def record(self, model=None):
        ind = 0
        if model is not None:
            self.tracker[-1]["l2(w-w0)"].append(0)
            self.tracker[-1]["l2(w)"].append(0)
            w = model.state_dict()
            for k in w.keys():
                self.tracker[-1]["l2(w-w0)"][-1]+=torch.sum((w[k]-self.w0[k])**2).item()
                self.tracker[-1]["l2(w)"][-1] += torch.sum(w[k]**2).item()

            self.tracker[-1]["l2(w-w0)"][-1] = np.sqrt(self.tracker[-1]["l2(w-w0)"][-1])
            self.tracker[-1]["l2(w)"][-1] = np.sqrt(self.tracker[-1]["l2(w)"][-1])

        for group in self.opt.param_groups:
            for param in group['params']:
                state = self.opt.state[param]

                p = param.data.cpu().detach().numpy().ravel()

                if param.grad is not None:
                    g = param.grad.data.cpu().detach().numpy().ravel()
                else:
                    g = np.ones_like(p)
                    
                if 'alpha_ratio' in state:
                    a = state['alpha_ratio'].cpu().detach().numpy().ravel()
                else:
                    a = np.ones_like(p)

                if 'feature_step' in state:
                    f = state['feature_step'].cpu().detach().numpy().ravel()
                elif 'step' in state:
                    f = np.ones_like(p) * state['step']
                else:
                    f = np.ones_like(p)

                if 'lr' in state:
                    lr = state['lr'].cpu().detach().numpy().ravel()
                else:
                    lr = np.ones_like(p) * group['lr']
                    
                for i, index in enumerate(self.index[param]):
                    self.tracker[ind]['grad'][i].append(g[index])
                    self.tracker[ind]['param'][i].append(p[index])                    
                    self.tracker[ind]['alpha_ratio'][i].append(a[index])
                    self.tracker[ind]['feature_step'][i].append(f[index])
                    self.tracker[ind]['lr'][i].append(lr[index])   
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

