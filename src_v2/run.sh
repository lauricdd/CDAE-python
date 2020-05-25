#!/usr/bin/env bash


# CDAE
for itr0 in 0 1 2 3 4
    do
        for itr1 in politic_old politic_new
            do
            python3 main.py --model_name=CDAE --data=${itr1} --encoder_method=SDAE --test_fold=${itr0} --corruption_level=0.4 \
            --f_act=Sigmoid --g_act=Sigmoid --hidden_neuron=50 --train_epoch=2000
            done
done