#!/bin/bash
GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU

python inference.py \
    --dictionary_path /scratch3/users/ylchou109/Pei_heng_weight/codebook/Codebook-4096Kmeans-1922_1_4year.pth \
    --checkpoint_path /scratch3/users/ylchou109/data_transfer/CWA_Stage3_LoRA_MR_2021_redo_fix_data_1year_6.pth \
    --data_dir /scratch3/users/ylchou109/regrid_cwa \
    --predict_time 2024010100


# instructions:
# --dictionary_path: path to the dictionary, e.g. ./NuWa/Dictionary-4year.pth
# --checkpoint_path: path to the checkpoint, e.g. ./NuWa/NuWa-1year.pth
# --data_dir: root dir path of the dataset
# --predict_time 'List of years, e.g. --years 2021 2022 2023 / 202101 202102 202103'


