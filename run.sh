#!/usr/bin/bash
set -e

WANDB_MODE="offline" /home/ubuntu/miniconda3/envs/pytorch/bin/python simulation.py --cfg_path configs/vit_config/fed_vit_various2h_iid_noagg.yaml
WANDB_MODE="offline" /home/ubuntu/miniconda3/envs/pytorch/bin/python simulation.py --cfg_path configs/vit_config/fed_vit_various2h_iid.yaml
WANDB_MODE="offline" /home/ubuntu/miniconda3/envs/pytorch/bin/python simulation.py --cfg_path configs/vit_config/fed_vit_various2h_noniid_noagg.yaml
WANDB_MODE="offline" /home/ubuntu/miniconda3/envs/pytorch/bin/python simulation.py --cfg_path configs/vit_config/fed_vit_various2h_noniid.yaml