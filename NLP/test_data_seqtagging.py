import argparse
from data.seq_tagging.data_loader import load

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, default=0)

    parser.add_argument("--is_debug_mode", default=0, type=int, help="is_debug_mode")

    # Data related
    # TODO: list all dataset names:
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikiner",
        metavar="N",
        help="dataset used for training",
    )

    parser.add_argument(
        "--data_file_path",
        type=str,
        default="../fednlp_data/data_files/wikiner_data.h5",
        help="data h5 file path",
    )

    parser.add_argument(
        "--partition_file_path",
        type=str,
        default="../fednlp_data/partition_files/wikiner_partition.h5",
        help="partition h5 file path",
    )

    parser.add_argument(
        "--partition_method", type=str, default="uniform", help="partition method"
    )

    # Model related
    parser.add_argument(
        "--model_type",
        type=str,
        default="bert",
        metavar="N",
        help="transformer model type",
    )

    parser.add_argument(
        "--model_class",
        type=str,
        default="transformer",
        metavar="N",
        help="model class",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-uncased",
        metavar="N",
        help="transformer model name",
    )
    parser.add_argument(
        "--do_lower_case",
        type=bool,
        default=True,
        metavar="N",
        help="transformer model name",
    )

    # Learning related
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        metavar="N",
        help="input batch size for training (default: 8)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        metavar="N",
        help="input batch size for evaluation (default: 8)",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        metavar="N",
        help="maximum sequence length (default: 128)",
    )

    parser.add_argument(
        "--n_gpu", type=int, default=1, metavar="EP", help="how many gpus will be used "
    )

    parser.add_argument(
        "--fp16", default=False, action="store_true", help="if enable fp16 for training"
    )
    parser.add_argument(
        "--manual_seed", type=int, default=42, metavar="N", help="random seed"
    )

    # IO related
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        metavar="N",
        help="path to save the trained results and ckpts",
    )

    # Federated Learning related
    parser.add_argument(
        "--federated_optimizer",
        type=str,
        default="FedAvg",
        help="Algorithm list: FedAvg; FedOPT; FedProx ",
    )

    parser.add_argument(
        "--backend", type=str, default="MPI", help="Backend for Server and Client"
    )

    parser.add_argument(
        "--comm_round",
        type=int,
        default=10,
        help="how many round of communications we shoud use",
    )

    parser.add_argument(
        "--is_mobile",
        type=int,
        default=1,
        help="whether the program is running on the FedML-Mobile server side",
    )

    parser.add_argument(
        "--client_num_in_total",
        type=int,
        default=-1,
        metavar="NN",
        help="number of clients in a distributed cluster",
    )

    parser.add_argument(
        "--client_num_per_round",
        type=int,
        default=4,
        metavar="NN",
        help="number of workers",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        metavar="EP",
        help="how many epochs will be trained locally",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        metavar="EP",
        help="how many steps for accumulate the loss.",
    )

    parser.add_argument(
        "--client_optimizer",
        type=str,
        default="adam",
        help="Optimizer used on the client. This field can be the name of any subclass of the torch Opimizer class.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate on the client (default: 0.001)",
    )

    parser.add_argument(
        "--weight_decay", type=float, default=0, metavar="N", help="L2 penalty"
    )

    parser.add_argument(
        "--clip_grad_norm", type=int, default=0, metavar="N", help="L2 penalty"
    )

    parser.add_argument(
        "--server_optimizer",
        type=str,
        default="sgd",
        help="Optimizer used on the server. This field can be the name of any subclass of the torch Opimizer class.",
    )

    parser.add_argument(
        "--server_lr",
        type=float,
        default=0.1,
        help="server learning rate (default: 0.001)",
    )

    parser.add_argument(
        "--server_momentum", type=float, default=0, help="server momentum (default: 0)"
    )

    parser.add_argument(
        "--fedprox_mu", type=float, default=1, help="server momentum (default: 1)"
    )

    parser.add_argument(
        "--evaluate_during_training",
        default=False,
        metavar="EP",
        help="the frequency of the evaluation during training",
    )

    parser.add_argument(
        "--evaluate_during_training_steps",
        type=int,
        default=100,
        metavar="EP",
        help="the frequency of the evaluation during training",
    )

    parser.add_argument(
        "--frequency_of_the_test",
        type=int,
        default=1,
        help="the frequency of the algorithms",
    )

    # GPU device management
    parser.add_argument(
        "--gpu_mapping_file",
        type=str,
        default="gpu_mapping.yaml",
        help="the gpu utilization file for servers and clients. If there is no \
                    gpu_util_file, gpu will not be used.",
    )

    parser.add_argument(
        "--gpu_mapping_key",
        type=str,
        default="mapping_default",
        help="the key in gpu utilization file",
    )

    parser.add_argument("--ci", type=int, default=0, help="CI")

    # cached related
    parser.add_argument(
        "--reprocess_input_data", action="store_true", help="whether generate features"
    )

    # freeze related
    parser.add_argument(
        "--freeze_layers", type=str, default="", metavar="N", help="freeze which layers"
    )
    args = parser.parse_args()
    dataset, class_num = load(args)
