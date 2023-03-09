#!/bin/bash

# 1. Evaluation of Seq2Seq Models (w/ & w/t Factorization)
# 1.1 NOT Using Expectation
python eval.py --model_type Seq2Seq --use_dens --use_spd_all --use_spd_truck --use_spd_pv
python eval.py --model_type Seq2Seq --use_spd_all --use_spd_truck --use_spd_pv
python eval.py --model_type Seq2Seq --use_dens --use_spd_all 
python eval.py --model_type Seq2Seq

# 1.2 Using Expectation
python eval.py --model_type Seq2Seq --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
python eval.py --model_type Seq2Seq --use_spd_all --use_spd_truck --use_spd_pv --use_expectation
python eval.py --model_type Seq2Seq --use_dens --use_spd_all --use_expectation
python eval.py --model_type Seq2Seq --use_expectation