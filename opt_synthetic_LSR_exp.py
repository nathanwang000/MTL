import torch, tqdm
import torch.nn as nn
import numpy as np
from lib.optimizer import Diff, Avrng, AdamVR, MomentumCurvature
from torch.utils.data import TensorDataset, DataLoader
from sklearn.externals import joblib
from lib.utils import random_string

def train(net, loader, full_loader, optimizer, x_star, niters=5000):
    losses = []
    errors = []
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

            p = list(net.parameters())[0].cpu().detach().numpy()
            errors.append(np.linalg.norm(p - x_star))

            for x, y in full_loader:
                x, y = x.cuda(), y.cuda()
                o = net(x)
                l = nn.MSELoss()(o.view(-1), y)
            losses.append(l.item())            
            
    return errors, losses

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

def main():
    kappas = [1, 10**2, 10**4, 10**8]
    nrepeat = 5
    d = 50
    n = 300
    batchsizes = [int(n), int(n/2), int(n/6)] # 50, 150, 300
    optimizations = ['Avrng', 'MomentumCurvature', 'torch.optim.Adam', 'torch.optim.SGD']
    lrs = [10, 1, 0.1, 0.01, 0.001]

    for i in range(nrepeat): # 5
        for k in kappas: # 4
            A, y, x_star = generate_data(k, d, n)      
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
                            'opt': opt,
                            'lr': lr
                        }

                        print(res)
                        net = nn.Linear(d, 1, bias=False)
                        optimizer = eval(opt)(net.parameters(), lr=lr)
                        name = opt.split('.')[-1]
                        errors, losses = train(net, loader, full_loader,
                                               optimizer, x_star=x_star)
                        res['errors'] = errors
                        res['losses'] = losses
                        joblib.dump(res, "{}/{}.pkl".format('synthetic_data_results/LSR',
                                                            random_string(5)))

if __name__ == '__main__':
    torch.set_num_threads(1)    
    main()
