#!/bin/bash

# 1. Evaluation of Baseline (Seq2Seq without Factorization)
python eval.py --task base --use_density --use_speed --use_truck_spd --use_pv_spd
python eval.py --task base --use_speed --use_truck_spd --use_pv_spd
python eval.py --task base --use_density --use_speed 
python eval.py --task base

# 2. Evaluation of Our Model (Seq2Seq with Factorization, using expection)
python eval.py --task finetune --use_density --use_speed --use_truck_spd --use_pv_spd --use_expectation
python eval.py --task finetune --use_speed --use_truck_spd --use_pv_spd --use_expectation
python eval.py --task finetune --use_density --use_speed --use_expectation
python eval.py --task finetune --use_expectation

# 3. Evaluation of Our Model (Seq2Seq with Factorization, not using expection)
python eval.py --task finetune --use_ density --use_speed --use_truck_spd --use_pv_spd 
python eval.py --task finetune --use_speed --use_truck_spd --use_pv_spd 
python eval.py --task finetune --use_density --use_speed
python eval.py --task finetune 