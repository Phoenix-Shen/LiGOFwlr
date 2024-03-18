#!/usr/bin/bash
set -e

# dir="configs/abaltion/only_large/c20_agnews"
# for filename in $(ls $dir)
# do
#    echo "Running simulation with config $dir/$filename"
#    python simulation.py --cfg_path $dir/$filename
# done

# dir="configs/abaltion/various_client_number/c30_agnews"
# for filename in $(ls $dir)
# do
#    echo "Running simulation with config $dir/$filename"
#    python simulation.py --cfg_path $dir/$filename
# done

# dir="configs/abaltion/various_client_number/c40_agnews"
# for filename in $(ls $dir)
# do
#    echo "Running simulation with config $dir/$filename"
#    python simulation.py --cfg_path $dir/$filename
# done

dir="configs/abaltion/various_client_number/c50_agnews"
for filename in $(ls $dir)
do
   echo "Running simulation with config $dir/$filename"
   python simulation.py --cfg_path $dir/$filename
done
