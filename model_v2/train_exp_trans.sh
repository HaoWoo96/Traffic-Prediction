#!/bin/bash

# Baseline - No Factorization
python train.py --model_type Trans --task no_fact --use_dens --use_spd_all --use_spd_truck --use_spd_pv --dim_hidden 448
python train.py --model_type Trans --task no_fact --use_spd_all --use_spd_truck --use_spd_pv --dim_hidden 448
python train.py --model_type Trans --task no_fact --use_dens --use_spd_all --dim_hidden 448
python train.py --model_type Trans --task no_fact --use_spd_all --dim_hidden 448
# # python train.py --model_type Trans --task no_fact


# Baseline - Naive Combination
# python train.py --model_type Trans --task naive --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
# python train.py --model_type Trans --task naive --use_dens --use_spd_all --use_spd_truck --use_spd_pv


python train.py --model_type Trans --task LR --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
python train.py --model_type Trans --task rec --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
python train.py --model_type Trans --task nonrec --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation --num_epochs 121 --lr 0.0001
python train.py --model_type Trans --task finetune --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
python train.py --model_type Trans --task finetune --use_dens --use_spd_all --use_spd_truck --use_spd_pv 

python train.py --model_type Trans --task LR --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
python train.py --model_type Trans --task rec  --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
python train.py --model_type Trans --task nonrec --use_spd_all --use_spd_truck --use_spd_pv --use_expectation --num_epochs 121 --lr 0.0001
python train.py --model_type Trans --task finetune --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
python train.py --model_type Trans --task finetune --use_spd_all --use_spd_truck --use_spd_pv

python train.py --model_type Trans --task LR --use_dens --use_spd_all --use_expectation
python train.py --model_type Trans --task rec --use_dens --use_spd_all --use_expectation
python train.py --model_type Trans --task nonrec --use_dens --use_spd_all --use_expectation --num_epochs 121 --lr 0.0001
python train.py --model_type Trans --task finetune --use_dens --use_spd_all --use_expectation
python train.py --model_type Trans --task finetune --use_dens --use_spd_all

python train.py --model_type Trans --task LR --use_spd_all --use_expectation
python train.py --model_type Trans --task rec --use_spd_all --use_expectation
python train.py --model_type Trans --task nonrec --use_spd_all --use_expectation --num_epochs 121 --lr 0.0001
python train.py --model_type Trans --task finetune --use_spd_all --use_expectation
python train.py --model_type Trans --task finetune --use_spd_all

# python train.py --model_type Trans --task LR --use_expectation
# python train.py --model_type Trans --task rec --use_expectation
# python train.py --model_type Trans --task nonrec --use_expectation
# python train.py --model_type Trans  --task finetune --use_expectation
# python train.py --model_type Trans  --task finetune