python ./SubpopBench-main/subpopbench/scripts/download.py \
--datasets "waterbirds" "metashift" "celeba" \
--data_path "../../data/new" \
--download True


python Ladder/src/codebase/SubpopBench-main/subpopbench/scripts/download.py \
--datasets "metashift" \
--data_path "Ladder/data" \
--download False


python download.py \
--datasets "mimic_cxr" \
--data_path "Ladder/data" \
--download True

python download.py \
--datasets "chexpert" \
--data_path "/ocean/projects/asc170022p/shared/Data/chestXRayDatasets/StanfordCheXpert" \
--download False


python download.py \
--datasets "celeba" \
--data_path "Ladder/data/celeba/data/archive"


#hyparams vit-oai
#{'resnet18': False, 'nonlinear_classifier': False, 'group_balanced': False,
#'pretrained': True, 'lr': 0.001, 'weight_decay': 0.0001, 'optimizer': 'sgd',
#'last_layer_dropout': 0.0, 'batch_size': 108, 'image_arch': 'vit_clip_oai',
#'text_arch': 'bert-base-uncased', 'steps': 5001}
python Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "GroupDRO" \
       --dataset "Waterbirds" \
       --train_attr no \
       --data_dir "Ladder/data" \
       --output_dir "Ladder/out/Waterbirds" \
       --output_folder_name "vit_sup_in1k" \
       --image_arch "vit_sup_in1k"


python Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "ERM" \
       --dataset "Waterbirds" \
       --train_attr no \
       --data_dir "Ladder/data" \
       --output_dir "Ladder/out/Waterbirds" \
       --output_folder_name "vit_sup_in21k" \
       --image_arch "vit_sup_in21k"

#{'resnet18': False, 'nonlinear_classifier': False, 'group_balanced': False,
#'pretrained': True, 'lr': 0.001, 'weight_decay': 0.0001, 'optimizer': 'sgd',
#'last_layer_dropout': 0.0, 'batch_size': 108, 'image_arch': 'vit_dino_in1k', 'text_arch': 'bert-base-uncased',
#'steps': 5001}
python Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "ERM" \
       --dataset "Waterbirds" \
       --train_attr no \
       --data_dir "Ladder/data" \
       --output_dir "Ladder/out/Waterbirds" \
       --output_folder_name "vit_dino_in1k" \
       --image_arch "vit_dino_in1k"


python Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "ERM" \
       --dataset "Waterbirds" \
       --train_attr no \
       --data_dir "Ladder/data" \
       --output_dir "Ladder/out/Waterbirds" \
       --output_folder_name "resnet_sup_in1k" \
       --image_arch "resnet_sup_in1k"

python Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "ERM" \
       --dataset "Waterbirds" \
       --train_attr no \
       --data_dir "Ladder/data" \
       --output_dir "Ladder/out/Waterbirds" \
       --output_folder_name "vit_sup_in1k" \
       --image_arch "vit_sup_in1k"

#python Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
#       --seed 0 \
#       --algorithm "DFR" \
#       --dataset "Waterbirds" \
#       --train_attr yes \
#       --stage1_folder "Ladder/out/Waterbirds/vit_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}" \
#       --data_dir "Ladder/data" \
#       --output_dir "Ladder/out/Waterbirds" \
#       --output_folder_name "vit_sup_in1k" \
#       --image_arch "vit_sup_in1k"

python Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "DFR" \
       --dataset "Waterbirds" \
       --train_attr yes \
       --data_dir "Ladder/data" \
       --output_dir "Ladder/out/Waterbirds" \
       --output_folder_name "vit_sup_in1k" \
       --image_arch "vit_sup_in1k"

python Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "ERM" \
       --dataset "CheXpertNoFinding" \
       --train_attr no \
       --data_dir "Ladder/data/chexpert-nofinding" \
       --output_dir "Ladder/out/CheXpertNoFinding" \
       --output_folder_name "resnet_sup_in1k" \
       --image_arch "resnet_sup_in1k"


#=======>>>> hparams <<<===========
#<class 'dict'>
#{'resnet18': False, 'nonlinear_classifier': False,
#'group_balanced': False, 'pretrained': True,
# 'lr': 0.001, 'weight_decay': 0.0001,
#  'optimizer': 'sgd', 'last_layer_dropout': 0.0,
#  'batch_size': 108, 'image_arch': 'vit_sup_in21k',
#   'text_arch': 'bert-base-uncased', 'steps': 30001}
python Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "ERM" \
       --dataset "CelebA" \
       --train_attr no \
       --data_dir "Ladder/data" \
       --output_dir "Ladder/out/CelebA" \
       --output_folder_name "vit_sup_in21k" \
       --image_arch "vit_sup_in21k"


#=======>>>> hparams <<<===========
#<class 'dict'>
#{'resnet18': False, 'nonlinear_classifier': False, 'group_balanced': False,
#'pretrained': True, 'lr': 0.001, 'weight_decay': 0.0001,
# 'optimizer': 'sgd', 'last_layer_dropout': 0.0, 'batch_size': 108,
# 'image_arch': 'vit_dino_in1k', 'text_arch': 'bert-base-uncased', 'steps': 30001}
python Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "ERM" \
       --dataset "CelebA" \
       --train_attr no \
       --data_dir "Ladder/data" \
       --output_dir "Ladder/out/CelebA" \
       --output_folder_name "vit_dino_in1k" \
       --image_arch "vit_dino_in1k"

python Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "ERM" \
       --dataset "CelebA" \
       --train_attr no \
       --data_dir "Ladder/data" \
       --output_dir "Ladder/out/CelebA" \
       --output_folder_name "resnet_sup_in1k" \
       --image_arch "resnet_sup_in1k"


python Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "ERM" \
       --dataset "CelebA" \
       --train_attr no \
       --data_dir "Ladder/data" \
       --output_dir "Ladder/out/CelebA" \
       --output_folder_name "vit_sup_in1k" \
       --image_arch "vit_sup_in1k"


python Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "ERM" \
       --dataset "MetaShift" \
       --train_attr no \
       --data_dir "Ladder/data" \
       --output_dir "Ladder/out/MetaShift" \
       --output_folder_name "vit_sup_in1k" \
       --image_arch "vit_sup_in1k"

python Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 0 \
       --algorithm "ERM" \
       --dataset "MetaShift" \
       --train_attr no \
       --data_dir "Ladder/data" \
       --output_dir "Ladder/out/MetaShift" \
       --output_folder_name "resnet_sup_in1k" \
       --image_arch "resnet_sup_in1k"

python Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 1 \
       --algorithm "ERM" \
       --dataset "MetaShift" \
       --train_attr no \
       --data_dir "Ladder/data" \
       --output_dir "Ladder/out/MetaShift" \
       --output_folder_name "resnet_sup_in1k" \
       --image_arch "resnet_sup_in1k"

python Ladder/src/codebase/SubpopBench-main/subpopbench/train.py \
       --seed 2 \
       --algorithm "ERM" \
       --dataset "MetaShift" \
       --train_attr no \
       --data_dir "Ladder/data" \
       --output_dir "Ladder/out/MetaShift" \
       --output_folder_name "resnet_sup_in1k" \
       --image_arch "resnet_sup_in1k"