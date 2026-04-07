#!/bin/bash

GPU=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU

python inference.py \
    --test_checkpoint_path /scratch3/users/ylchou109/Allen_Backup_for_new_Global_MR/Allen_Taiwan-Aurora-Foundation-Model/Stage3/weights/cwa_ignore_missing_regrid/CWA_Stage3_LoRA_MR_2021_redo_fix_data_1mon/CWA_Stage3_LoRA_MR_2021_redo_fix_data_1mon_20.pth \
    --test_time 2024010100 \
    --root_dir /scratch3/users/phchiang/NuWa \
    --data_dir /scratch3/users/ylchou109/regrid_cwa \
    --codebook_path /scratch3/users/ylchou109/Pei_heng_weight/codebook/Codebook-4096Kmeans-1922_1_4year.pth
