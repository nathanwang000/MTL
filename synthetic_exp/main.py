from Optimizer.lib.utils import OptPath
from Optimizer.lib.utils import gen_quadratic_loss, gen_quadratic_data, gen_criteria
import Optimizer.lib.optimizer as optimizers
import numpy as np
import torch
from itertools import product
from sklearn.externals import joblib
import argparse
import os
torch.set_num_threads(1)
parser = argparse.ArgumentParser(description="optimizer")
parser.add_argument('-max_iter', type=int,
                    help='number of iterations to run', default=3000)
parser.add_argument('-k', type=int, default=0,
                    help='condition number in log scale')
parser.add_argument('-lm', type=int, default=0,
                    help='lambda max in log scale')
parser.add_argument('-d', type=int, help='dimension', default=100)
parser.add_argument('-s', type=int, help='seed', default=0)
parser.add_argument('-savedir', type=str, help='dir to save', default='result:0.1')
args = parser.parse_args()

os.system('mkdir -p {}'.format(args.savedir))
opt_path = OptPath(max_iter=args.max_iter)
k = 10 ** args.k
lambda_max = 10 ** args.lm
d = args.d
seed = args.s
n = 300
bs = 1

# generate the problem
np.random.seed(seed)
# criteria, Q, Lambda = gen_quadratic_loss(d, lambda_max / k, lambda_max,
#                                          logscale=True)
# x0 = np.random.uniform(2, 10, d)

Q, Lambda, X, y = gen_quadratic_data(n, d, lambda_max / k, lambda_max, logscale=True)
criteria, x_star = gen_criteria(X, y)
x0  = np.random.randn(d)

# set optimizers
opt_settings = {
    'Adam': {'opt': torch.optim.Adam,  'lr': 0.1},
    'AdaSGD': {'opt': optimizers.AdaSGD,  'lr': 0.01, 'momentum': 0.9},
    'SGD': {'opt': torch.optim.SGD, 'lr': 0.01, 'momentum': 0.9},        
    'OptSGD': {'opt': torch.optim.SGD, 'lr': bs / lambda_max},

    # 'SGDbetas': {'opt': optimizers.SGDbetas,  'lr': 0.1},
    # 'SGDbeta': {'opt': optimizers.SGDbetas,  'lr':  1*bs/lambda_max, 'momentum': 0.9},
    # 'AdjustSGD': {'opt': torch.optim.SGD, 'lr': 1e-10, 'momentum': 0.9,
    #               'sgd_adjust': True}        
    
}

for name in opt_settings:
    opt_path.get_path(criteria, x0, bs=bs, **opt_settings[name])
    to_save = opt_settings[name]
    to_save['loss'] = [np.abs(criteria(torch.from_numpy(x).view(1, -1).float(),
                                       bs=n).item() -
                              criteria(torch.from_numpy(x_star).view(1, -1).float(),
                                       bs=n).item()) + 1e-10
                       for x in opt_path.x_path]
    joblib.dump(to_save, '{}/{}:{}:{}:{}'.format(
        args.savedir, name, k, lambda_max, seed))


