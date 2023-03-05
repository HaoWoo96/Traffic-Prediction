#!/bin/bash

# python train.py --task base --use_dens --use_spd_all --use_spd_truck --use_spd_pv
# python train.py --task base --use_spd_all --use_spd_truck --use_spd_pv
# python train.py --task base --use_dens --use_spd_all 
# python train.py --task base



# python train.py --task LR --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
# python train.py --task rec --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
# python train.py --task nonrec --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
# python train.py --task finetune --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation


# python train.py --task LR --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
# python train.py --task rec  --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
# python train.py --task nonrec --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
# python train.py --task finetune --use_spd_all --use_spd_truck --use_spd_pv --use_expectation

# python train.py --task LR --use_dens --use_spd_all --use_expectation
# python train.py --task rec --use_dens --use_spd_all --use_expectation
# python train.py --task nonrec --use_dens --use_spd_all --use_expectation
# python train.py --task finetune --use_dens --use_spd_all --use_expectation

# python train.py --task LR --use_expectation
# python train.py --task rec --use_expectation
# python train.py --task nonrec --use_expectation
# python train.py --task finetune --use_expectation



# python train.py --task LR --use_dens --use_spd_all --use_spd_truck --use_spd_pv 
# python train.py --task rec --use_dens --use_spd_all --use_spd_truck --use_spd_pv 
# python train.py --task nonrec --use_dens --use_spd_all --use_spd_truck --use_spd_pv 
python train.py --task finetune --use_dens --use_spd_all --use_spd_truck --use_spd_pv 

# python train.py --task LR --use_spd_all --use_spd_truck --use_spd_pv 
# python train.py --task rec  --use_spd_all --use_spd_truck --use_spd_pv 
# python train.py --task nonrec --use_spd_all --use_spd_truck --use_spd_pv 
# python train.py --task finetune --use_spd_all --use_spd_truck --use_spd_pv 

# python train.py --task LR --use_dens --use_spd_all 
# python train.py --task rec --use_dens --use_spd_all 
# python train.py --task nonrec --use_dens --use_spd_all 
# python train.py --task finetune --use_dens --use_spd_all

# python train.py --task LR 
# python train.py --task rec 
# python train.py --task nonrec 
# python train.py --task finetune 