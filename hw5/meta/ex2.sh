#!/usr/bin/env bash
python train_policy.py 'pm' --exp_name ex2_ff --history 60 --discount 0.90 -lr 5e-4 -n 60 &
python train_policy.py 'pm' --exp_name ex2_rec --history 60 --discount 0.90 -lr 5e-4 -n 60 --recurrent &
python train_policy.py 'pm' --exp_name ex3_inter1 --history 60 --discount 0.90 -lr 5e-4 -n 60 --interval 1