from Optimizer.lib.utils import OptPath
from Optimizer.lib.utils import gen_quadratic_loss
import Optimizer.lib.optimizer as optimizers
import numpy as np
import torch
from itertools import product
from sklearn.externals import joblib
import argparse
import os

parser = argparse.ArgumentParser(description="optimizer")
parser.add_argument('-max_iter', type=int,
                    help='number of iterations to run', default=3000)
parser.add_argument('-k', type=int, default=0,
                    help='condition number in log scale')
parser.add_argument('-lm', type=int, default=0,
                    help='lambda max in log scale')
parser.add_argument('-d', type=int, help='dimension', default=100)
parser.add_argument('-s', type=int, help='seed', default=0)
parser.add_argument('-savedir', type=str, help='dir to save', default='result:0.01')
args = parser.parse_args()

os.system('mkdir -p {}'.format(args.savedir))
opt_path = OptPath(max_iter=args.max_iter)
k = 10 ** args.k
lambda_max = 10 ** args.lm
d = args.d
seed = args.s

# generate the problem
np.random.seed(seed)
criteria, Q, Lambda = gen_quadratic_loss(d, lambda_max / k, lambda_max,
                                         logscale=True)
x0 = np.random.uniform(2, 10, d)

# set optimizers
opt_settings = {
    'Adam': {'opt': torch.optim.Adam,  'lr': 0.01},
    'AdaSGD': {'opt': optimizers.AdaSGD,  'lr': 0.01, 'momentum': 0.9},
    # 'SGD': {'opt': torch.optim.SGD, 'lr': 0.01, 'momentum': 0.9},        
    # 'OptSGD': {'opt': torch.optim.SGD, 'lr': 1/lambda_max, 'momentum': 0.9},
    # 'AdjustSGD': {'opt': torch.optim.SGD, 'lr': 1e-10, 'momentum': 0.9,
    #               'sgd_adjust': True}        
}

for name in opt_settings:
    opt_path.get_path(criteria, x0, **opt_settings[name])
    to_save = opt_settings[name]
    to_save['loss'] = opt_path.get_loss()
    joblib.dump(to_save, '{}/{}:{}:{}:{}'.format(
        args.savedir, name, k, lambda_max, seed))


