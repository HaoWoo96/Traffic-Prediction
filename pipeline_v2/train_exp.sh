#!/bin/bash

# python train.py --task LR --use_density --use_speed --use_truck_spd --use_pv_spd --use_expectation
# python train.py --task rec --use_density --use_speed --use_truck_spd --use_pv_spd --use_expectation
# python train.py --task nonrec --use_density --use_speed --use_truck_spd --use_pv_spd --use_expectation
# python train.py --task finetune --use_density --use_speed --use_truck_spd --use_pv_spd --use_expectation


# python train.py --task LR --use_speed --use_truck_spd --use_pv_spd --use_expectation
# python train.py --task rec  --use_speed --use_truck_spd --use_pv_spd --use_expectation
# python train.py --task nonrec --use_speed --use_truck_spd --use_pv_spd --use_expectation
# python train.py --task finetune --use_speed --use_truck_spd --use_pv_spd --use_expectation

# python train.py --task LR --use_density --use_speed --use_expectation
# python train.py --task rec --use_density --use_speed --use_expectation
# python train.py --task nonrec --use_density --use_speed --use_expectation
# python train.py --task finetune --use_density --use_speed --use_expectation

# python train.py --task LR --use_expectation
# python train.py --task rec --use_expectation
# python train.py --task nonrec --use_expectation
# python train.py --task finetune --use_expectation

# python train.py --task base --use_density --use_speed --use_truck_spd --use_pv_spd
python train.py --task base --use_speed --use_truck_spd --use_pv_spd
python train.py --task base --use_density --use_speed 
python train.py --task base

python train.py --task LR --use_density --use_speed --use_truck_spd --use_pv_spd 
python train.py --task rec --use_density --use_speed --use_truck_spd --use_pv_spd 
python train.py --task nonrec --use_density --use_speed --use_truck_spd --use_pv_spd 
python train.py --task finetune --use_ density --use_speed --use_truck_spd --use_pv_spd 

python train.py --task LR --use_speed --use_truck_spd --use_pv_spd 
python train.py --task rec  --use_speed --use_truck_spd --use_pv_spd 
python train.py --task nonrec --use_speed --use_truck_spd --use_pv_spd 
python train.py --task finetune --use_speed --use_truck_spd --use_pv_spd 

python train.py --task LR --use_density --use_speed 
python train.py --task rec --use_density --use_speed 
python train.py --task nonrec --use_density --use_speed 
python train.py --task finetune --use_density --use_speed

python train.py --task LR 
python train.py --task rec 
python train.py --task nonrec 
python train.py --task finetune 