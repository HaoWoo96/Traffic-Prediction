#!/bin/bash

# Baseline - No Factorization
#python train.py --model_type Seq2Seq --task no_fact --use_dens --use_spd_all --use_spd_truck --use_spd_pv
#python train.py --model_type Seq2Seq --task no_fact --use_spd_all --use_spd_truck --use_spd_pv
#python train.py --model_type Seq2Seq --task no_fact --use_dens --use_spd_all 
#python train.py --model_type Seq2Seq --task no_fact --use_spd_all 
# python train.py --model_type Seq2Seq --task no_fact

# Baseline - Naive Combination
# python train.py --model_type Seq2Seq --task naive --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
# python train.py --model_type Seq2Seq --task naive --use_dens --use_spd_all --use_spd_truck --use_spd_pv

python train.py --model_type Seq2Seq --task LR --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation --num_epochs 301
python train.py --model_type Seq2Seq --task rec --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
python train.py --model_type Seq2Seq --task nonrec --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation --num_epochs 121 --lr 0.0001
python train.py --model_type Seq2Seq --task finetune --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
python train.py --model_type Seq2Seq --task finetune --use_dens --use_spd_all --use_spd_truck --use_spd_pv 

python train.py --model_type Seq2Seq --task LR --use_spd_all --use_spd_truck --use_spd_pv --use_expectation --num_epochs 301
python train.py --model_type Seq2Seq --task rec  --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
python train.py --model_type Seq2Seq --task nonrec --use_spd_all --use_spd_truck --use_spd_pv --use_expectation --num_epochs 121 --lr 0.0001
python train.py --model_type Seq2Seq --task finetune --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
python train.py --model_type Seq2Seq --task finetune --use_spd_all --use_spd_truck --use_spd_pv

python train.py --model_type Seq2Seq --task LR --use_dens --use_spd_all --use_expectation --num_epochs 301
python train.py --model_type Seq2Seq --task rec --use_dens --use_spd_all --use_expectation
python train.py --model_type Seq2Seq --task nonrec --use_dens --use_spd_all --use_expectation --num_epochs 121 --lr 0.0001
python train.py --model_type Seq2Seq --task finetune --use_dens --use_spd_all --use_expectation
python train.py --model_type Seq2Seq --task finetune --use_dens --use_spd_all

python train.py --model_type Seq2Seq --task LR --use_spd_all --use_expectation --num_epochs 301
python train.py --model_type Seq2Seq --task rec --use_spd_all --use_expectation
python train.py --model_type Seq2Seq --task nonrec --use_spd_all --use_expectation --lr 0.0001 --num_epochs 121
python train.py --model_type Seq2Seq --task finetune --use_spd_all --use_expectation
python train.py --model_type Seq2Seq --task finetune --use_spd_all

# python train.py --model_type Seq2Seq --task LR --use_expectation --num_epochs 301
# python train.py --model_type Seq2Seq --task rec --use_expectation
# python train.py --model_type Seq2Seq --task nonrec --use_expectation --lr 0.0001 --num_epochs 121
# python train.py --model_type Seq2Seq  --task finetune --use_expectation
# python train.py --model_type Seq2Seq  --task finetune
