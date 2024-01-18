#!/usr/bin/bash
set -e

pythonpath="/home/ubuntu/miniconda3/envs/fwlr/bin/python"
base_dir="configs/ablation/c20_cifar100"
#WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_iid.yaml
#WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_noniid.yaml
WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_iid_noagg.yaml
WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_noniid_noagg.yaml


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

# base_dir="configs/c20_flowers102"
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_iid.yaml
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_noniid.yaml
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_iid_noagg.yaml
# WANDB_MODE="offline" $pythonpath simulation.py --cfg_path $base_dir/fed_vit_various2h_noniid_noagg.yaml

