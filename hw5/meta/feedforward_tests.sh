#!/usr/bin/env bash

#python train_policy.py 'pm' --exp_name ex2_ff_h60 --history 60 --discount 0.90 -lr 5e-4 -n 60 &
#python train_policy.py 'pm' --exp_name ex2_ff_h20 --history 20 --discount 0.90 -lr 5e-4 -n 60 &
#python train_policy.py 'pm' --exp_name ex2_ff_h100 --history 100 --discount 0.90 -lr 5e-4 -n 60 &

python train_policy.py 'pm' --exp_name ex3_ff_h60_i1_lr1 --history 60 --discount 0.90 -lr 1e-4 -n 60 --interval 1 &
python train_policy.py 'pm' --exp_name ex3_ff_h60_i5_lr1 --history 60 --discount 0.90 -lr 1e-4 -n 60 --interval 5 &
python train_policy.py 'pm' --exp_name ex3_ff_h60_i5_lr5 --history 60 --discount 0.90 -lr 5e-4 -n 60 --interval 5 &

