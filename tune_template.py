import os
import signal
import atexit
import subprocess
from itertools import product
import time
import argparse

parser = argparse.ArgumentParser(description="param search")
parser.add_argument('-c', type=int, help='n concurrent process', default=None)
parser.add_argument('-gpus', type=int, nargs='+', default=[0,1,2,3,4,5,6,7],
                    help='gpus to use')

args = parser.parse_args()
print(args)

procs = []
n_concurrent_process = args.c if args.c is not None else len(args.gpus)

# change this part
weight_decays = [5e-6, 5e-7, 1e-5, 1e-6] #[5e-5, 5e-4, 5e-3, 5e-2, 5e-1]
gpu_id = 0
for wd in weight_decays:
    commands = ["python", "main.py", "--model", "resnet", "--optim", "adam",
                "--lr", "0.001", "--final_lr", "0.1", "--weight_decay", str(wd)]
    my_env = os.environ.copy()
    my_env['CUDA_VISIBLE_DEVICES'] = str(args.gpus[gpu_id])
    gpu_id = (gpu_id + 1) % len(args.gpus)
    procs.append(subprocess.Popen(commands, env=my_env))

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
        if p.poll() != None:
            pass
        elif p.pid is None:
            pass
        else:
            os.kill(p.pid, signal.SIGTERM)
        
atexit.register(kill_procs)

