#!/bin/bash

# Baseline

# python train.py --county TSMO --model_type GTrans --task no_fact --use_dens --use_spd_all --use_spd_truck --use_spd_pv --num_epochs 400 --batch_size 32 --num_STBlock 1 --lr 0.0002

# python train.py --county TSMO --model_type Trans --task no_fact --use_dens --use_spd_all --use_spd_truck --use_spd_pv --dim_hidden 1664 --num_epochs 900 --batch_size 128  # batch size 256 will lead to CUDA out of memory
python train.py --county TSMO --model_type Trans --task no_fact --use_dens --use_spd_all --use_spd_truck --use_spd_pv --dim_hidden 1664 --num_epochs 900 --batch_size 64 --lr 0.00015
# python train.py --county TSMO --model_type Trans --task no_fact --use_dens --use_spd_all --use_spd_truck --use_spd_pv --dim_hidden 1664 --num_epochs 1800 --batch_size 128 --lr 0.0002 --load_checkpoint_epoch 800
# python train.py --county TSMO --model_type Trans --task no_fact --use_dens --use_spd_all --use_spd_truck --use_spd_pv --dim_hidden 1664 --num_epochs 1800 --batch_size 64 --lr 0.0002
# python train.py --county TSMO --model_type Trans --task no_fact --use_dens --use_spd_all --use_spd_truck --use_spd_pv --dim_hidden 2048 --num_epochs 900 --batch_size 64  # batch size 128 will lead to CUDA out of memory
# python train.py --county TSMO --model_type Seq2Seq --task no_fact --use_dens --use_spd_all --use_spd_truck --use_spd_pv --dim_hidden 1536 --num_epochs 900 --batch_size 128  # batch size 256 will lead to CUDA out of memory

# python train.py --county TSMO --model_type Trans --task no_fact --use_spd_all --use_spd_truck --use_spd_pv --dim_hidden 1664 --num_epochs 900 --batch_size 128
# python train.py --county TSMO --model_type Trans --task no_fact --use_dens --use_spd_all --dim_hidden 1664 --num_epochs 900 --batch_size 128
# python train.py --county TSMO --model_type Trans --task no_fact --use_spd_all --dim_hidden 1664 --num_epochs 900 --batch_size 128

# python train.py --county TSMO --model_type Seq2Seq --task no_fact --use_spd_all --use_spd_truck --use_spd_pv --dim_hidden 1536 --num_epochs 900 --batch_size 128
# python train.py --county TSMO --model_type Seq2Seq --task no_fact --use_dens --use_spd_all --dim_hidden 1536 --num_epochs 900 --batch_size 128
# python train.py --county TSMO --model_type Seq2Seq --task no_fact --use_spd_all --dim_hidden 1536 --num_epochs 900 --batch_size 128


# Finetune Pipeline
# python train.py --county TSMO --model_type Seq2Seq --task LR --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation

# --- Finished ---
# python train.py --county TSMO --model_type Trans --task LR --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation --dim_hidden 1024  --batch_size 256
# python train.py --county TSMO --model_type Trans --task rec --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation --dim_hidden 1024  --batch_size 256
# python train.py --county TSMO --model_type Trans --task nonrec --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation --dim_hidden 1024  --batch_size 256
# python train.py --county TSMO --model_type Trans --task finetune --use_dens --use_spd_all --use_spd_truck --use_spd_pv --use_expectation --dim_hidden 1024  --batch_size 256

