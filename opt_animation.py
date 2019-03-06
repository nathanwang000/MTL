import numpy as np
import torch, os
from lib.utils import random_string
import argparse
from sklearn.externals import joblib
from MTL_OPT.lib.utils import AverageMeter, OptRecorder
import MTL_OPT.lib.optimizer as optimizers

parser = argparse.ArgumentParser(description="opt")
parser.add_argument('-o', type=str,
                    help='optimizer', default='optimizers.AlphaDiff(1,1)')
parser.add_argument('-lr', type=float,
                    help='learning rate', default=0.003)
parser.add_argument('-g', type=float,
                    help='grad noise', default=0.5)
parser.add_argument('-s', type=str,
                    help='save directory', default='synthetic_data_results/animation')
args = parser.parse_args()
train_losses = []
torch.set_num_threads(1)

# Hyper-parameters
num_epochs = 30000
grad_noise = args.g
learning_rate = args.lr

class Segment(object):
    
    def __init__(self, b, f):
        # assume start from 0
        self.b = b
        self.f = f
    
    def connectRight(self, other):
        # piece together line segments
        # assume self is on the left, other is on the right
        
        def f(x):
            f1b = self.f(self.b)
            f2b = other.f(self.b)

            if x <= self.b:
                return self.f(x)
            else:
                return other.f(x) - (f2b - f1b)
            
        return Segment(other.b, f)
        
    def plot(self, b=None, c='b'):
        x = np.linspace(0, b or self.b, 100)
        plt.plot(x, [self.f(x_) for x_ in x], label='f(x)', c=c)
        
    def plot_grad(self, b=None, c='b'):
        b = b or self.b
        grads = []
        xs = np.linspace(0, b, 100)
        for x in xs:
            x = torch.nn.Parameter(torch.Tensor([x]))
            y = s.forward(x)
            y.backward()
            grads.append(x.grad.item())
        plt.plot(xs, grads, label='grad', c=c)
        
    def forward(self, x):
        return self.f(x)
            
s1 = Segment(10, lambda x: -0.5*x + 10) # A
s2 = Segment(20, lambda x: -5*x) # D
s3 = Segment(30, lambda x: -1*x + 10) # A
s4 = Segment(100, lambda x: ((x-42.5)/5)**2) # E
s = s1.connectRight(s2).connectRight(s3).connectRight(s4)

p = torch.nn.Parameter(torch.Tensor([0])) # start from 0
if '(' in args.o:
    opt = args.o
    alpha_index = opt.find('(')
    alphas = eval(opt[alpha_index:])
    optimizer = eval(opt[:alpha_index])([p], lr=learning_rate,
                                        alphas=alphas)
else:
    optimizer = eval(args.o)([p], lr=learning_rate)
opt_recorder = OptRecorder(optimizer)    

for epoch in range(num_epochs):

    # sample input
    delta = np.random.choice([-1, -1, -1, 3])
    def closure():
        optimizer.zero_grad()
        l = s.forward(p) # p is parameter, s is the curve
        l.backward()
        p.grad.data.add_(grad_noise * delta)
        return l

    l = optimizer.step(closure)

    if epoch % max(int(num_epochs/5000), 1) == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch+1, num_epochs, l.item()))
        opt_recorder.record()            
        train_losses.append(l.item())


os.system('mkdir -p {}'.format(args.s))
name =  "{}/{}-{}^{:.2f}^{}".format(args.s,
                                    args.o.split('.')[-1],
                                    args.lr,
                                    l.item(),
                                    random_string(5))

joblib.dump(train_losses, name + ".train_losses")
joblib.dump(opt_recorder.tracker, name + ".opt_track")
        


