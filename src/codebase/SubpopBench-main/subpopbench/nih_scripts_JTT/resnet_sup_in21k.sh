#!/bin/sh
#SBATCH --output=src/psc_logs/subpopbench/nih-JTT/nih-resnet_sup_in21k-%j.out

pwd
hostname
date

CURRENT=$(date +"%Y-%m-%d_%T")

echo $CURRENT

slurm_output_train1=src/psc_logs/subpopbench/nih-JTT/nih-resnet_sup_in21k-$CURRENT.out

echo "Save image reps"
source /ocean/projects/asc170022p/shg121/anaconda3/etc/profile.d/conda.sh

conda activate /restricted/projectnb/batmanlab/shawn24/breast_clip_rtx_6000


# redo
python src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "JTT" \
       --dataset "NIH_dataset" \
       --train_attr yes \
       --data_dir "data" \
       --output_dir "out/NIH_Cxrclip" \
       --output_folder_name "resnet_sup_in21k" \
       --image_arch "resnet_sup_in21k" \
       --es_metric overall:AUROC --use_es >$slurm_output_train1



