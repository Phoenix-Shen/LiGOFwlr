#!/usr/bin/bash
set -e

python simulation.py --cfg_path configs/tc/c20_agnews/agnews_c20_iid.yaml
python simulation.py --cfg_path configs/tc/c20_agnews/agnews_c20_iid_noagg.yaml
