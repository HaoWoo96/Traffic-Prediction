#!/bin/bash

# Baseline
# python train.py --model_type Trans --task no_fact --use_dens --use_spd_all --use_spd_truck --use_spd_pv --dim_hidden 448 --num_epochs 1800
# python train.py --model_type Trans --task no_fact --use_spd_all --use_spd_truck --use_spd_pv --dim_hidden 448 --num_epochs 1800
# python train.py --model_type Trans --task no_fact --use_dens --use_spd_all --dim_hidden 448 --num_epochs 1800
# python train.py --model_type Trans --task no_fact --use_spd_all --dim_hidden 448 --num_epochs 1800

# Finetune from Scratch
# python train.py --model_type Trans --task finetune_from_scratch --use_dens --use_spd_all --use_spd_truck --use_spd_pv --num_epochs 1800
# python train.py --model_type Trans --task finetune_from_scratch --use_spd_all --use_spd_truck --use_spd_pv --use_expectation --num_epochs 1200
# python train.py --model_type Trans --task finetune_from_scratch --use_spd_all --use_spd_truck --use_spd_pv --num_epochs 1200
# python train.py --model_type Trans --task finetune_from_scratch --use_dens --use_spd_all --use_expectation --num_epochs 1200
# python train.py --model_type Trans --task finetune_from_scratch --use_dens --use_spd_all --num_epochs 1200
# python train.py --model_type Trans --task finetune_from_scratch --use_spd_all --use_expectation --num_epochs 1200
# python train.py --model_type Trans --task finetune_from_scratch --use_spd_all --num_epochs 1200

# Finetune Pipeline
#'''
#Finished:

# python train.py --model_type Trans --task LR --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
# python train.py --model_type Trans --task rec --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
# python train.py --model_type Trans --task nonrec --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation --num_epochs 121
# python train.py --model_type Trans --task finetune --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation --num_epochs 1800
# python train.py --model_type Trans --task finetune --use_dens --use_spd_all --use_spd_truck --use_spd_pv --num_epochs 1800
#'''

# python train.py --model_type Trans --task LR --use_spd_all --use_expectation
# python train.py --model_type Trans --task rec --use_spd_all --use_expectation
# python train.py --model_type Trans --task nonrec --use_spd_all --use_expectation --num_epochs 121 
# python train.py --model_type Trans --task finetune --use_spd_all --use_expectation --num_epochs 1800
# python train.py --model_type Trans --task finetune --use_spd_all --num_epochs 1800


# python train.py --model_type Trans --task LR --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
# python train.py --model_type Trans --task rec  --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
# python train.py --model_type Trans --task nonrec --use_spd_all --use_spd_truck --use_spd_pv --use_expectation --num_epochs 121
# python train.py --model_type Trans --task finetune --use_spd_all --use_spd_truck --use_spd_pv --use_expectation --num_epochs 1800
# python train.py --model_type Trans --task finetune --use_spd_all --use_spd_truck --use_spd_pv --num_epochs 1800

# python train.py --model_type Trans --task LR --use_dens --use_spd_all --use_expectation
python train.py --model_type Trans --task rec --use_dens --use_spd_all --use_expectation
# python train.py --model_type Trans --task nonrec --use_dens --use_spd_all --use_expectation --num_epochs 121 
python train.py --model_type Trans --task finetune --use_dens --use_spd_all --use_expectation  --num_epochs 1800
python train.py --model_type Trans --task finetune --use_dens --use_spd_all --num_epochs 1800