#!/bin/bash
for grad_noise in {0,0.5,1.0,5.0};
do
    for lr in {1e-3,1e-4,100,10,1,0.1,0.01};
    do
	python opt_animation.py -o 'optimizers.AdaBound' -lr $lr -g $grad_noise
	python opt_animation.py -o 'optimizers.CrossBound' -lr $lr -g $grad_noise
	python opt_animation.py -o 'optimizers.CrossAdaBound' -lr $lr -g $grad_noise
	python opt_animation.py -o 'optimizers.Swats' -lr $lr -g $grad_noise

	python opt_animation.py -o 'optimizers.AdamC1(1,1)' -lr $lr -g $grad_noise
	python opt_animation.py -o 'optimizers.AdamC2(1,1)' -lr $lr -g $grad_noise
	python opt_animation.py -o 'optimizers.AlphaDiff(1,1)' -lr $lr -g $grad_noise
	python opt_animation.py -o 'optimizers.AlphaAdam(1,1)' -lr $lr -g $grad_noise
	python opt_animation.py -o 'optimizers.AlphaSGD(1,1)' -lr $lr -g $grad_noise
	python opt_animation.py -o 'optimizers.AlphaDiff(1,0)' -lr $lr -g $grad_noise
	python opt_animation.py -o 'optimizers.AlphaAdam(1,0)' -lr $lr -g $grad_noise
	python opt_animation.py -o 'torch.optim.SGD' -lr $lr -g $grad_noise
	python opt_animation.py -o 'optimizers.AlphaDiff(0,1)' -lr $lr -g $grad_noise
	python opt_animation.py -o 'optimizers.AlphaAdam(0,1)' -lr $lr -g $grad_noise
    done
done
