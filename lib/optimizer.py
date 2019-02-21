import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
import math 

class Sign(Optimizer):
    '''
    only use the sign of gradient
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(Sign, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                #     # for book keeping
                #     state['grad_history'] = []
                # state['grad_history'].append(grad.cpu().clone())

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                bias_correction1 = 1 - beta1 ** state['step']
                step_size = group['lr'] / bias_correction1

                p.data.add_(-step_size, torch.sign(exp_avg))

        return loss

class NormalizedCurvature(Optimizer):
    '''
    (gt - g{t-1})^2
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(NormalizedCurvature, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # keep last gradient
                    state['last_grad'] = torch.zeros_like(p.data)
                    state['last_w'] = torch.zeros_like(p.data)
                    state['w_diff'] = torch.ones_like(p.data)
                    state['grad_diff'] = torch.ones_like(p.data) 

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                if state['step'] == 1:
                    w_diff = torch.ones_like(p.data)  # don't use diff at first step
                else:
                    w_diff = torch.abs(p.data - state['last_w']) + group['eps']  

                diff = ((grad - state['last_grad']) / w_diff)**2
                # state['w_diff'].mul_(beta2).add_(1-beta2, w_diff)
                # grad_diff = torch.abs(grad - state['last_grad'])
                # state['grad_diff'].mul_(beta2).add_(1-beta2, grad_diff)
                # pct_grad_diff = grad_diff / state['grad_diff']
                # diff = pct_grad_diff**2
                # pct_wdiff = w_diff / state['w_diff']
                #diff = (pct_grad_diff / pct_wdiff)**2

                
                
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                denom = exp_avg_sq.sqrt().add_(group['eps']) 

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # step_size = 0
                #numer = torch.min(step_size / denom, 0.1 * torch.ones_like(exp_avg))
                numer = step_size / denom
                #p.data.addcdiv_(-step_size, exp_avg, denom)
                state['last_grad'] = grad.data.clone()
                state['last_w'] = p.data.clone()                                
                p.data.add_(-numer * exp_avg)

        return loss
    
class MomentumCurvature(Optimizer):
    '''
    (gt - g{t-1})^2
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, max_lr=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, max_lr=max_lr)
        super(MomentumCurvature, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # keep last gradient
                    state['last_grad'] = torch.zeros_like(p.data)
                    state['last_w'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                if state['step'] == 1:
                    bias_correction = 1 # first time no bias reduction
                else:
                    bias_correction = 1 - beta1 ** (state['step']-1)
                mhat = exp_avg / bias_correction
                
                #denom = torch.abs(p.data - state['last_w']) + group['eps']
                #diff = ((grad - state['last_grad']) / denom)**2
                diff = (grad - state['last_grad'])**2                
                # print(torch.min(p.data - state['last_w']))
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                denom = exp_avg_sq.sqrt().add_(group['eps']) 

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if group['max_lr']:
                    numer = torch.min(step_size / denom,
                                      group['max_lr'] * torch.ones_like(exp_avg))
                else:
                    numer = step_size / denom
                #p.data.addcdiv_(-step_size, exp_avg, denom)
                state['last_grad'] = grad.data.clone()
                state['last_w'] = p.data.clone()                                
                p.data.add_(-numer * exp_avg)

        return loss

class MomentumCurvature2(Optimizer):
    '''
    abs(gt - g{t-1})
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, max_lr=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, max_lr=max_lr)
        super(MomentumCurvature2, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # keep last gradient
                    state['last_grad'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                if state['step'] == 1:
                    bias_correction = 1 # first time no bias reduction
                else:
                    bias_correction = 1 - beta1 ** (state['step']-1)
                mhat = exp_avg / bias_correction
                
                diff = torch.abs(grad - state['last_grad'])
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                denom = exp_avg_sq.add_(group['eps'])  # no square

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * bias_correction2 / bias_correction1

                if group['max_lr']:
                    numer = torch.min(step_size / denom,
                                      group['max_lr'] * torch.ones_like(exp_avg))
                else:
                    numer = step_size / denom
                #p.data.addcdiv_(-step_size, exp_avg, denom)
                state['last_grad'] = grad.data.clone()
                p.data.add_(-numer * exp_avg)

        return loss
    
class RK4(Optimizer):
    
    def __init__(self, params, lr=required, weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(RK4, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RK4, self).__setstate__(state)

    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        
        for i in range(4):
            loss = closure() # recompute gradient
                
            for group in self.param_groups:

                for p in group['params']:
                    if p.grad is None:
                        continue
                
                    # save k1 to k4 to param_state
                    param_state = self.state[p]
                    if 'k' not in param_state:
                        param_state['k'] = [0] * 4 # list of size 4
                    param_state['k'][i] = p.grad.data
                    
                    coefs = [1, 0.5, 0.5, 1]
                    # undo last update
                    if i != 0: # add back gradient
                        p.data.add_(group['lr'] * coefs[i], param_state['k'][i-1])
                    # update
                    if i != 3: # intermediate update
                        p.data.add_(-group['lr'] * coefs[i+1], param_state['k'][i])
                    else: # real update
                        k1, k2, k3, k4 = param_state['k']
                        p.data.add_(-group['lr'] / 6, k1 + 2*k2 + 2*k3 + k4) 
        
        return loss
    

class DoublingRK4(Optimizer):
    
    '''
    take 1 full step and 2 half step and compare the difference,
    not a proper implementation of Doubling RK4 as 
    '''
    def __init__(self, params, lr=required, weight_decay=0, tol=1e-7):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay, tol=tol)
        super(DoublingRK4, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RK4, self).__setstate__(state)

    def step_scale(self, closure, scale):
        loss = None
        
        for i in range(4):
            loss = closure() # recompute gradient
                
            for group in self.param_groups:

                for p in group['params']:
                    if p.grad is None:
                        continue
                
                    # save k1 to k4 to param_state
                    param_state = self.state[p]
                    if 'k' not in param_state:
                        param_state['k'] = [0] * 4 # list of size 4
                    param_state['k'][i] = p.grad.data
                    
                    coefs = [1, 0.5, 0.5, 1]
                    # undo last update
                    if i != 0: # add back gradient
                        p.data.add_(group['lr'] * coefs[i] * scale, param_state['k'][i-1])
                    # update
                    if i != 3: # intermediate update
                        p.data.add_(-group['lr'] * coefs[i+1] * scale, param_state['k'][i])
                    else: # real update
                        k1, k2, k3, k4 = param_state['k']
                        p.data.add_(-group['lr'] / 6 * scale, k1 + 2*k2 + 2*k3 + k4)

                        # save this update: to be compared later
                        savename = 'update{}'.format(scale)
                        if savename not in param_state:
                            param_state[savename] = []
                        param_state[savename].append(-group['lr'] / 6 * scale * (k1 + 2*k2 + 2*k3 + k4))
                        
        return loss

    def undo_step(self, nsteps, scale):
        '''
        nsteps is how many steps back to undo
        '''
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                param_state = self.state[p]
                savename = 'update{}'.format(scale)
                for i, change in enumerate(param_state[savename][::-1]):
                    if i > nsteps:
                        break
                    p.data.add_(-change)
        
    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # take 1 large step
        self.step_scale(closure, 2) 
        # undo the step
        self.undo_step(nsteps=1, scale=2)
        
        # take 2 small steps
        self.step_scale(closure, 1)
        loss = self.step_scale(closure, 1)
        
        # compare steps
        for group in self.param_groups:
            tol = group['tol'] # tolerence get from defaults

            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_state = self.state[p]
                
                y2 = sum(param_state['update2'])
                y1 = sum(param_state['update1'])
                #print(y1.shape, y2.shape)
                
                # find max deviation
                delta = torch.max(torch.abs(y2 - y1))

                factor = (tol / delta)**(1/5)
                tmplr = group['lr'] * factor
                group['lr'] = min(max(tmplr, 1e-8), 0.1) # this seems inefficient
                #print(group['lr'], delta)
                #print(group['lr'])
                
                # restore update to none
                param_state['update1'] = []
                param_state['update2'] = []

        return loss

class Avrng(Optimizer):
    '''
    adaptive variance reduced & (curvature) normalized gradient
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, max_lr=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, max_lr=max_lr)
        super(Avrng, self).__init__(params, defaults)

    def rewind(self):
        '''
        rewind 1 step back and calculate old gradient
        '''
        for group in self.param_groups:

            for p in group['params']:
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # last change
                    state['delta'] = torch.zeros_like(p.data)

                p.data.add_(-state['delta']) # reverse change

    def forward(self):
        '''
        forward 1 step back and calculate old gradient
        '''
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                p.data.add_(state['delta']) # forward change

                # save old grad
                state['old_grad'] = p.grad.data.clone()
                
    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # last time step
        self.rewind()
        closure()
        # go back to now and save grad to old_grad
        self.forward()
        loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
               
                # Decay the first and second moment running average coefficient
                if state['step'] == 1:
                    old_grad = torch.zeros_like(state['old_grad'])
                else:
                    old_grad = state['old_grad']
                
                # variance reduced gradient:
                diff = (grad - old_grad)**2 # capture curvature, not variance
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                denom = exp_avg_sq.sqrt().add_(group['eps']) # note: denom is per term, doesnt sound

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # note: sqrt bias_correction2 is here because denominator is sqrt for bias_correction2 as well
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # save delta
                if group['max_lr']:
                    numer = torch.min(step_size / denom,
                                      group['max_lr'] * torch.ones_like(exp_avg))
                else:
                    numer = step_size / denom
                delta = -numer * exp_avg
                state['delta'] = delta
                p.data.add_(delta)
                #p.data.addcdiv_(-step_size, exp_avg, denom)
                
        return [loss, loss] # two times

class AdamVR(Optimizer):
    '''
    adaptive variance reduced & (curvature) normalized gradient
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, max_lr=1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, max_lr=max_lr)
        super(AdamVR, self).__init__(params, defaults)

    def rewind(self):
        '''
        rewind 1 step back and calculate old gradient
        '''
        for group in self.param_groups:

            for p in group['params']:
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # last change
                    state['delta'] = torch.zeros_like(p.data)

                p.data.add_(-state['delta']) # reverse change

    def forward(self):
        '''
        forward 1 step back and calculate old gradient
        '''
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                p.data.add_(state['delta']) # forward change

                # save old grad
                state['old_grad'] = p.grad.data.clone()
                
    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # last time step
        self.rewind()
        closure()
        # go back to now and save grad to old_grad
        self.forward()
        loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
               
                # Decay the first and second moment running average coefficient
                if state['step'] == 1:
                    bias_correction = 1 # first time no bias reduction
                else:
                    bias_correction = 1 - beta1 ** (state['step']-1)
                mhat = exp_avg / bias_correction
                
                # variance reduced gradient:
                gradhat = grad - state['old_grad'] + mhat
                    
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, gradhat**2)
                denom = exp_avg_sq.sqrt().add_(group['eps']) # note: denom is per term, doesnt sound

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # note: sqrt bias_correction2 is here because denominator is sqrt for bias_correction2 as well
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # save delta
                numer = torch.min(step_size / denom, group['max_lr'] * torch.ones_like(exp_avg))
                delta = -numer * exp_avg
                state['delta'] = delta
                p.data.add_(delta)
                #p.data.addcdiv_(-step_size, exp_avg, denom)
                
        return [loss, loss] # two times
    
class Adam2(Optimizer):
    '''
    do 2 Adam update
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, max_lr=1):
        self.adam = torch.optim.Adam(params, lr=lr, betas=betas, eps=eps)

    def zero_grad(self):
        self.adam.zero_grad()
        
    def step(self, closure):
        loss1 = self.adam.step(closure)
        loss2 = self.adam.step(closure)
        return [loss1, loss2]
        
class Avrng2(Optimizer):
    '''
    adaptive variance reduced & (curvature) normalized gradient
    do twice, without wasting any update
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, max_lr=1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, max_lr=max_lr)
        super(Avrng2, self).__init__(params, defaults)

    def update1(self):
        '''
        rewind 1 step back and calculate old gradient
        '''
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                if state['step'] == 1:
                    bias_correction = 1 # first time no bias reduction
                else:
                    bias_correction = 1 - beta1 ** (state['step']-1)
                mhat = exp_avg / bias_correction
                
                diff = (grad - mhat)**2 # capture curvature, not variance
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                denom = exp_avg_sq.sqrt().add_(group['eps']) # note: denom is per term, doesnt sound

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # save delta
                #numer = torch.min(step_size / denom,
                #                  group['max_lr'] * torch.ones_like(exp_avg))
                #p.data.add_(-numer * exp_avg)
                p.data.addcdiv_(-step_size, exp_avg, denom)
                
                # save old grad
                state['old_grad'] = p.grad.data.clone()
                
    def step(self, closure):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss1 = closure()
        self.update1()

        loss2 = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
               
                # Decay the first and second moment running average coefficient
                if state['step'] == 1:
                    bias_correction = 1 # first time no bias reduction
                else:
                    bias_correction = 1 - beta1 ** (state['step']-1)
                mhat = exp_avg / bias_correction
                
                # variance reduced gradient:
                gradhat = grad - state['old_grad'] + mhat
                    
                diff = (gradhat - mhat)**2 # capture curvature, not variance
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                denom = exp_avg_sq.sqrt().add_(group['eps']) # note: denom is per term, doesnt sound

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # note: sqrt bias_correction2 is here because denominator is sqrt for bias_correction2 as well
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # save delta
                # numer = torch.min(step_size / denom,
                #                   group['max_lr'] * torch.ones_like(exp_avg))
                # p.data.add_(-numer * exp_avg)
                p.data.addcdiv_(-step_size, exp_avg, denom)
                
        return [loss1, loss2]
    
class Diff(Optimizer):
    '''
    use V to record difference between estimates instead!
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(Diff, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                if state['step'] == 1:
                    bias_correction = 1 # first time no bias reduction
                else:
                    bias_correction = 1 - beta1 ** (state['step']-1)
                mhat = exp_avg / bias_correction
                
                diff = (grad - mhat)**2 # this is the only change from Adam!!!!
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                denom = exp_avg_sq.sqrt().add_(group['eps']) 

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # note: sqrt bias_correction2 is here because denominator is sqrt for bias_correction2 as well
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

class DiffUnbiased(Optimizer):
    '''
    use V to record difference between estimates instead!
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(DiffUnbiased, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    # note: use actual value
                    state['exp_avg'] = grad.clone()
                    # Exponential moving average of squared difference in gradient values
                    # note: use one step look ahead: that is not update for the first
                    # step
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                diff = (grad - exp_avg)**2 # this is the only change from Adam!!!!   
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                if state['step'] == 2:
                    state['exp_avg_sq'] = diff.clone()
                    exp_avg_sq = state['exp_avg_sq']
                else:
                    exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                    
                # note: denom is per term, doesnt sound
                if state['step'] <= 1: # accumulate for a few steps
                    denom = torch.ones_like(p.data)
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                p.data.addcdiv_(-group['lr'], exp_avg, denom)

        return loss

class DiffUnbiasedBounded(Optimizer):
    '''
    use V to record difference between estimates instead!
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, upper_bound=1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, upper_bound=upper_bound)
        super(DiffUnbiasedBounded, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    # note: use actual value
                    state['exp_avg'] = grad.clone()
                    # Exponential moving average of squared difference in gradient values
                    # note: use one step look ahead: that is not update for the first
                    # step
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                diff = (grad - exp_avg)**2 # this is the only change from Adam!!!!   
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                if state['step'] == 2:
                    state['exp_avg_sq'] = diff.clone()
                    exp_avg_sq = state['exp_avg_sq']
                else:
                    exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                    
                # note: denom is per term, doesnt sound
                if state['step'] <= 1: # accumulate for a few steps
                    denom = torch.ones_like(p.data)
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                numer = torch.min(group['lr'] / denom,
                                  group['upper_bound'] * torch.ones_like(exp_avg))
                p.data.add_(-exp_avg * numer)

        return loss
    
class DiffMax(Optimizer):
    '''
    changed per parameter update to all together update
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super(DiffMax, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared difference in gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                diff = (grad - exp_avg)**2 # this is the only change from Adam!!!! 
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).add_(1 - beta2, diff)
                # only difference from Diff1 is to make denom a scaler
                denom = torch.max(exp_avg_sq.sqrt()).add_(group['eps']) # only difference from Diff

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

class AdamUnbiased(Optimizer):
    
    '''
    changed unbiasing operation from Adam
    '''
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamUnbiased, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad #torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad**2#torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                #bias_correction1 = 1 - beta1 ** state['step']
                #bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr']# * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

