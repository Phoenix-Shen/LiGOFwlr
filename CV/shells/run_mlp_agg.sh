#!/bin/bash
set -e
#cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Variables
device="cuda:0"
cfg_path="configs/mlp_config/fed_mlp_various2h_noniid.yaml"

echo "Starting server"
python server.py --cfg_path $cfg_path&
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 0 19`; do
    echo "Starting client $i"
    python client.py --cfg_path $cfg_path --partition $i --device $device&
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

# Variables
device="cuda:0"
cfg_path="configs/mlp_config/fed_mlp_various2h_noniid_noagg.yaml"

echo "Starting server"
python server.py --cfg_path $cfg_path&
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 0 19`; do
    echo "Starting client $i"
    python client.py --cfg_path $cfg_path --partition $i --device $device&
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
