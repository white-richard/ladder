#!/bin/sh
#SBATCH --output=src/psc_logs/subpopbench/waterbirds_vit/LfF-%j.out

pwd
hostname
date

CURRENT=$(date +"%Y-%m-%d_%T")

echo $CURRENT

slurm_output_train1=src/psc_logs/subpopbench/waterbirds_vit/LfF-$CURRENT.out

echo "Save image reps"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh

conda activate /restricted/projectnb/batmanlab/shawn24/breast_clip_rtx_6000

python src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "LfF" \
       --dataset "Waterbirds" \
       --train_attr yes \
       --data_dir "data" \
       --output_dir "out/Waterbirds" \
       --output_folder_name "vit_sup_in1k" \
       --image_arch "vit_sup_in1k" >$slurm_output_train1