from Optimizer.lib.utils import OptPath
from Optimizer.lib.utils import gen_quadratic_loss
import Optimizer.lib.optimizer as optimizers
import numpy as np
import torch
from itertools import product
from sklearn.externals import joblib
import argparse
import os
import signal
import atexit
import subprocess
import time

parser = argparse.ArgumentParser(description="optimizer")
parser.add_argument('-c', type=int, help='n concurrent process', default=10)
parser.add_argument('-max_iter', type=int,
                    help='number of iterations to run', default=3000)
parser.add_argument('-kappas', type=int, nargs='+', default=[0,2,4,6,8],
                    help='condition number in log scale')
parser.add_argument('-lambda_maxs', type=int, nargs='+', default=[0,2,4,6,8],
                    help='lambda max in log scale')
parser.add_argument('-d', type=int, help='dimension', default=100)
parser.add_argument('-r', type=int, help='number of repeat', default=30)
args = parser.parse_args()
print(args)

procs = []
n_concurrent_process = args.c

# change start
max_iter = args.max_iter
d = args.d
nruns = args.r

for k, lambda_max in product(args.kappas, args.lambda_maxs):
    for run in range(nruns):
        commands = ["python", "main.py", "-max_iter", str(max_iter),
                    "-k", str(k), "-lm", str(lambda_max), "-d", str(d),
                    "-s", str(run)]
        procs.append(subprocess.Popen(commands))

        # change end
        while True:
            new_procs = []
            while len(procs) > 0:
                p = procs.pop()
                if p.poll() == None: # active
                    new_procs.append(p)

            procs = new_procs
            if len(procs) >= n_concurrent_process:
                time.sleep(3)
            else:
                break # fetch next
                
for p in procs:
    p.wait()
    
def kill_procs():
    for p in procs:
        if p.poll() != None: # not active
            pass
        elif p.pid is None:
            pass
        else:
            os.kill(p.pid, signal.SIGTERM)
        
atexit.register(kill_procs)


