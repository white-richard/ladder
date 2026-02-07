#!/bin/sh
#SBATCH --output=Ladder/src/psc_logs/subpopbench/waterbirds_%j.out

pwd
hostname
date

CURRENT=$(date +"%Y-%m-%d_%T")

echo $CURRENT

slurm_output_train1=Ladder/src/psc_logs/subpopbench/waterbirds_seed0_$CURRENT.out
slurm_output_train2=Ladder/src/psc_logs/subpopbench/waterbirds_seed1_$CURRENT.out
slurm_output_train3=Ladder/src/psc_logs/subpopbench/waterbirds_seed2_$CURRENT.out

echo "Save image reps"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh

conda activate breast_clip_rtx_6000

python Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "GroupDRO" \
       --dataset "Waterbirds" \
       --train_attr yes \
       --data_dir "Ladder/data" \
       --output_dir "Ladder/out/Waterbirds" \
       --output_folder_name "vit_sup_in21k" \
       --image_arch "vit_sup_in21k"

python Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "GroupDRO" \
       --dataset "Waterbirds" \
       --train_attr yes \
       --data_dir "Ladder/data" \
       --output_dir "Ladder/out/Waterbirds" \
       --output_folder_name "vit_clip_oai" \
       --image_arch "vit_clip_oai"

python Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "GroupDRO" \
       --dataset "Waterbirds" \
       --train_attr yes \
       --data_dir "Ladder/data" \
       --output_dir "Ladder/out/Waterbirds" \
       --output_folder_name "vit_clip_laion" \
       --image_arch "vit_clip_laion"

python Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "GroupDRO" \
       --dataset "Waterbirds" \
       --train_attr yes \
       --data_dir "Ladder/data" \
       --output_dir "Ladder/out/Waterbirds" \
       --output_folder_name "vit_sup_swag" \
       --image_arch "vit_sup_swag"


python Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "GroupDRO" \
       --dataset "Waterbirds" \
       --train_attr yes \
       --data_dir "Ladder/data" \
       --output_dir "Ladder/out/Waterbirds" \
       --output_folder_name "vit_dino_in1k" \
       --image_arch "vit_dino_in1k"