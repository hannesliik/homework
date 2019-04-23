#!/usr/bin/env bash
#python train_policy.py 'pm' --exp_name ex2_rnn_h60 --history 60 --discount 0.90 -lr 5e-4 -n 60 --recurrent --interval 0 &
#python train_policy.py 'pm' --exp_name ex2_rnn_h20 --history 20 --discount 0.90 -lr 5e-4 -n 60 --recurrent --interval 0 &
#python train_policy.py 'pm' --exp_name ex2_rnn_h100 --history 100 --discount 0.90 -lr 5e-4 -n 60 --recurrent --interval 0 &

#python train_policy.py 'pm' --exp_name ex3_rnn_h60_i1 --history 60 --discount 0.90 -lr 5e-4 -n 60 --recurrent --interval 1 &
#python train_policy.py 'pm' --exp_name ex3_rnn_h20_i1 --history 20 --discount 0.90 -lr 5e-4 -n 60 --recurrent --interval 1 &
#python train_policy.py 'pm' --exp_name ex3_rnn_h100_i1 --history 100 --discount 0.90 -lr 5e-4 -n 60 --recurrent --interval 1 &

python train_policy.py 'pm' --exp_name ex3_rnn_h00_i5 --history 100 --discount 0.90 -lr 5e-4 -n 60 --recurrent --interval 5 &
python train_policy.py 'pm' --exp_name ex3_rnn_h100_i2 --history 100 --discount 0.90 -lr 5e-4 -n 60 --recurrent --interval 2 &
python train_policy.py 'pm' --exp_name ex3_rnn_h100_i0.5 --history 100 --discount 0.90 -lr 5e-4 -n 60 --recurrent --interval 0.5 &
