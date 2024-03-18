#!/usr/bin/bash
set -e

python simulation.py --cfg_path configs/tc/c20_agnews/agnews_c20_noniid.yaml
python simulation.py --cfg_path configs/tc/c20_agnews/agnews_c20_noniid_noagg.yaml
