#!/usr/bin/env bash
python train_ac_exploration_f18.py RoboschoolHalfCheetah-v1 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 --density_model none --exp_name HC_bc0 &
python train_ac_exploration_f18.py RoboschoolHalfCheetah-v1 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 --density_model ex2 -bc 0.001 -kl 0.1 -dlr 0.005 -dti 1000 --exp_name HC_bc0.001_kl0.1_dlr0.005_dti1000 &
python train_ac_exploration_f18.py RoboschoolHalfCheetah-v1 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 --density_model ex2 -bc 0.0001 -kl 0.1 -dlr 0.005 -dti 10000 --exp_name HC_bc0.0001_kl0.1_dlr0.005_dti10000 &