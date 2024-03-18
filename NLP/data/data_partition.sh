DATA_DIR=../fednlp_data

# python -m data.advanced_partition.niid_label \
# --client_number 5 \
# --data_file ${DATA_DIR}/data_files/20news_data.h5 \
# --partition_file ${DATA_DIR}/partition_files/20news_partition.h5 \
# --task_type text_classification \
# --skew_type label \
# --seed 42 \
# --kmeans_num 0  \
# --alpha 0.5
dataset_name="semeval_2010_task8"
task_type=sequence_tagging
client_numbers=(5 10 15 20 25 30)

for client_number in ${client_numbers[@]}
do
    python -m data.advanced_partition.niid_label \
    --client_number $client_number \
    --data_file ${DATA_DIR}/data_files/${dataset_name}_data.h5 \
    --partition_file ${DATA_DIR}/partition_files/${dataset_name}_partition.h5 \
    --task_type ${task_type} \
    --skew_type label \
    --seed 42 \
    --kmeans_num 10  \
    --alpha 0.5
done

# dataset_name="sst_2"
# task_type=text_classification
# client_numbers=(5 10 15 20 25 30)

# for client_number in ${client_numbers[@]}
# do
#     python -m data.advanced_partition.niid_quantity  \
#     --client_number $client_number  \
#     --data_file ${DATA_DIR}/data_files/${dataset_name}_data.h5  \
#     --partition_file ${DATA_DIR}/partition_files/${dataset_name}_partition.h5 \
#     --task_type ${task_type} \
#     --kmeans_num 0 \
#     --beta 5
# done

