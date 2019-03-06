import torch, tqdm, os
import torch.nn as nn
import numpy as np
from lib.optimizer import Diff
from lib.optimizer import AlphaAdam, AlphaDiff, AlphaSGD, AdamC1, AdamC2
from lib.optimizer import AdaBound, CrossBound, CrossAdaBound, Swats
from lib.utils import OptRecorder
from torch.utils.data import TensorDataset, DataLoader
from sklearn.externals import joblib
from lib.utils import random_string

def train(net, loader, full_loader, optimizer, niters=5000):
    losses = []
    opt_recorder = OptRecorder(optimizer)
    net.cuda()
    for _ in tqdm.tqdm(range(niters)):
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            def closure():
                optimizer.zero_grad()
                o = net(x)
                l = nn.MSELoss()(o.view(-1), y)
                l.backward()
            l = optimizer.step(closure)

        # record full loss
        for x, y in full_loader:
            x, y = x.cuda(), y.cuda()
            o = net(x)
            l = nn.MSELoss()(o.view(-1), y)
        losses.append(l.item())
        opt_recorder.record()
            
    return losses, opt_recorder.tracker

def generate_data(kappa=1, d=50, n=300):
    '''
    kappa: condition number
    d: feature dimension, keep this <= n so that condition number is finite
    n: number of observations
    '''
    assert d <= n, 'd <= n to make sure A.T.dot(A) has no 0 eigenvector'
    x_star = np.random.randn(d)
    x_star = x_star / np.linalg.norm(x_star)

    noise = np.random.randn(n)
    noise = noise / np.linalg.norm(noise) * 1e-3

    # data matrix
    A = 1/np.sqrt(n) * np.random.randn(n, d)
    U, S, Vh = np.linalg.svd(A)
    # np.allclose(U[:,:len(S)].dot(np.diag(S)).dot(Vh), A) gives True    

    # change the eigenvalues
    log_smin = -1 # that is mu = 0.1**2
    log_smax = np.log10(np.sqrt(kappa) * 10**log_smin)
    S = np.logspace(log_smin, log_smax, d)[::-1]

    A = U[:,:len(S)].dot(np.diag(S)).dot(Vh)
    y = A.dot(x_star) + noise

    ''' verify kappa
    import matplotlib.pyplot as plt
    _,S_,_ = np.linalg.svd(A.T.dot(A))
    print(np.max(S_), np.min(S_))
    '''

    # note: this is not axis aligned
    mu = (10**log_smin)**2
    L = kappa * mu
    print(r'\mu={:.2f}, L={:.2f}, \kappa={:.2f}'.format(mu, L, L/mu))
    return A, y, x_star

def get_data(kappa, d, n, run_number):
    savename = 'data/LSR/{}_{}_{}.pkl'.format(kappa, d, n, run_number)
    if os.path.exists(savename):
        return joblib.load(savename)
    else:
        data = generate_data(kappa, d, n)
        joblib.dump(data, savename)
        return data

def main():
    kappas = [1, 10**2, 10**4]
    nrepeat = 3
    d = 50
    n = 300
    batchsizes = [int(n), int(n/2), int(n/6)] # 50, 150, 300
    optimizations = [
        'Diff',
        'torch.optim.Adam',
        
        # test converting SGD
        # 'AdaBound',
        # 'CrossBound',
        # 'CrossAdaBound',
        # 'Swats',

        # test dominance
        # 'AdamC1(1,1)',
        # 'AdamC2(1,1)',                
        
        # 'AlphaDiff(1,1)', # same as MC
        # 'AlphaAdam(1,1)', # same as Adam
        # 'AlphaSGD(1,1)',                      
        
        # 'AlphaDiff(1,0)', # no var(dg)
        # 'AlphaAdam(1,0)', # no var(g)
        # 'torch.optim.SGD', # same as AlphaSGD(1,0) with 0 momentum
    
        # 'AlphaDiff(0,1)', # only var(dg)
        # 'AlphaAdam(0,1)', # only var(g), same as AlphaSGD(0,1)
    ]
    
    lrs = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001]

    for i in range(nrepeat): # 3
        for k in kappas: # 4
            A, y, x_star = get_data(k, d, n, i)
            for bs in batchsizes: # 3
                data = TensorDataset(torch.from_numpy(A).float(),
                                     torch.from_numpy(y).float())
                loader = DataLoader(data, batch_size=bs)
                full_loader = DataLoader(data, batch_size=n)                
                for opt in optimizations: # 3
                    for lr in lrs: # 5

                        res = {
                            'run': i,                            
                            'kappa': k,
                            'batch_size': bs,
                            'n': n,
                            'd': d, 
                            'opt': opt.split('.')[-1],
                            'lr': lr,
                        }

                        print(res)
                        net = nn.Linear(d, 1, bias=True)
                        if '(' in opt:
                            alpha_index = opt.find('(')
                            alphas = eval(opt[alpha_index:])
                            optimizer = eval(opt[:alpha_index])(net.parameters(),
                                             lr=lr, alphas=alphas)
                        else:
                            optimizer = eval(opt)(net.parameters(), lr=lr)

                        losses, opt_tracker = train(net, loader, full_loader, optimizer)
                        name = "{}/{}".format('synthetic_data_results/LSR',
                                              random_string(5))
                        joblib.dump(res, "{}.ind".format(name))
                        joblib.dump(losses, "{}.loss".format(name))
                        joblib.dump(opt_tracker, "{}.track".format(name))
                        
if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
