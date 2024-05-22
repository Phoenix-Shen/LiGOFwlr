#!/usr/bin/bash
set -e

pythonpath="/home/ubuntu/miniconda3/envs/pytorch/bin/python"
# base_dir="configs/ablation/c20_cifar100"
#WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_iid.yaml
#WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_noniid.yaml
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_iid_noagg.yaml
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_noniid_noagg.yaml


# base_dir="configs/c10_flowers102"
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_iid.yaml
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_noniid.yaml
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_iid_noagg.yaml
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_noniid_noagg.yaml

# base_dir="configs/c20_cifar10"
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_iid.yaml
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_noniid.yaml
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_iid_noagg.yaml
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_noniid_noagg.yaml



# base_dir="configs/ablation/various_client_number/c30_cifar100"
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_iid.yaml
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_noniid.yaml
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_iid_noagg.yaml
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_noniid_noagg.yaml

# base_dir="configs/ablation/various_client_number/c40_cifar100"
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_iid.yaml
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_noniid.yaml
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_iid_noagg.yaml
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_noniid_noagg.yaml

# base_dir="configs/ablation/various_client_number/c50_cifar100"
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_iid.yaml
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_noniid.yaml
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_iid_noagg.yaml
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_noniid_noagg.yaml


WANDB_MODE="offline" $pythonpath simulation.py --cfg_path configs/additional_experiments/different_model_hetero2_noagg.yaml
WANDB_MODE="offline" $pythonpath simulation.py --cfg_path configs/additional_experiments/different_model_hetero1_noagg.yaml
WANDB_MODE="offline" $pythonpath simulation.py --cfg_path configs/additional_experiments/different_lr1.yaml
WANDB_MODE="offline" $pythonpath simulation.py --cfg_path configs/additional_experiments/different_lr2.yaml
WANDB_MODE="offline" $pythonpath simulation.py --cfg_path configs/additional_experiments/different_lr1_noagg.yaml
WANDB_MODE="offline" $pythonpath simulation.py --cfg_path configs/additional_experiments/different_lr2_noagg.yaml