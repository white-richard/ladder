#!/bin/sh
#SBATCH --output=Ladder/src/psc_logs/subpopbench/celebA_vit/LfF-%j.out

pwd
hostname
date

CURRENT=$(date +"%Y-%m-%d_%T")

echo $CURRENT

slurm_output_train1=Ladder/src/psc_logs/subpopbench/celebA_vit/LfF-$CURRENT.out

echo "Save image reps"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh

conda activate breast_clip_rtx_6000

python Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "LfF" \
       --dataset "CelebA" \
       --train_attr yes \
       --data_dir "Ladder/data" \
       --output_dir "Ladder/out/CelebA/LfF" \
       --output_folder_name "vit_sup_in1k" \
       --image_arch "vit_sup_in1k" >$slurm_output_train1