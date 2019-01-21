import torch, math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import warnings

class SharedBottom(nn.Module):

    def __init__(self, l=4):
        '''
        l: number of layers
        '''
        super(SharedBottom, self).__init__()
        self.l = l

        if l == 0: # keep output 16
            a = 16
        elif l == 1: # keep parameter comparable
            a = int(np.round((14672 - 16) / 116)) 
        elif l == 2: # keep output 16 and parameters comparable
            a = int(np.round((14672 - 16 - 16*8*2) / 116))
        else: # keep output 16 and parameters comparable
            a = int(np.round((-116 + np.sqrt(116**2 + 4*(l-2)*
                                             (14672 - 16 - 16*8*2)))
                             / (2*(l-2))))

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
        if l == 0: # keep ouput 16
            a = 16 # effective 1 layer
        elif l == 1: # keep parameters comparable 
            a = int(np.round((14672/2-8) / 108))
        elif l == 2: # keep output 16 and parameters comparable
            a = int(np.round((14672/2-8-16*8) / 116))
        else: # keep output 16 and parameters comparable
            c = 14672/2 - 8 - 16*8
            a = int(np.round((-116 + np.sqrt(116**2 + 4*(l-2)*c)) / (2 * (l-2))))

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

