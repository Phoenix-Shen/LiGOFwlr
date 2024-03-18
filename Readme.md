# Fed-Grow: Federating to Grow Transformers for Resource-Constrained Users without Model Sharing

Here is the code repo of our paper: Fed-Grow: Federating to Grow Transformers for Resource-Constrained Users without Model Sharing.

## 1. Code Structure

```bash
.
├── CV # Code for CV tasks
│   ├── client.py # Client 
│   ├── configs # Configs for all experiments
│   ├── data.py # data processing 
│   ├── datasets # store the datasets
│   ├── engine.py # train/eval
│   ├── environment.yaml
│   ├── logs # logs
│   ├── models # store the neural networks
│   ├── run.sh 
│   ├── server.py # the server
│   ├── shells # shells for run experiments
│   ├── simulation.py # run simulations (we mainly use this file to run experiments)
│   ├── strategy.py 
│   ├── test.ipynb
│   ├── tutorial
│   ├── utils.py
│   └── wandb # wandb files
├── NLP
│   ├── client.py
│   ├── configs
│   ├── ... # Essentially the same as CV
└── Readme.md

```
- The entire framework is based on [Flower](https://flower.ai)
- For cv tasks, our code is mainly based on facebook's [deit](https://github.com/facebookresearch/deit) repo.
- For nlp tasks, our code is mainly based on [FedNLP](https://github.com/FedML-AI/FedML/tree/3f3e7c7aafe9fac51343110406bc0a9e7d097f16/python/examples/federate/prebuilt_jobs/fednlp) repo.

## 2. Preparation

- 1. Create a new anaconda env.

    ```bash
    conda create -n fedgrow python=3.9
    conda activate fedgrow
    ```

- 2. Install dependencies
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers evaluate datasets flwr[simulation] h5py wandb timm scikit-learn seqeval
    # The above are the main dependencies, for other missing dependencies, you can download via pip manually.
    ```

- 3. Prepare data
    - For cv tasks, we use `torchvision.datasets`, so you just need to wait for the dataset to finish downloading automatically.
    - For nlp tasks, it is a bit complicated.
        - Prepare BertTokenizer into `NLP/pretrained_models/bert_tokenizer` directory (Chinese users can't connect to Huggingface, so we have to download it manually), or you can modify all `BertTokenizer.from_pretrained(
            "pretrained_models/bert_tokenizer"
        )` according to your actual situation.
        - Prepare data, you can refer to `NLP/data/README.md` to download data manually.
        - Split data, you can refer to `NLP/data/advanced_partition/README.md` to split the data manually. Finally, you need to split the appropriate dataset based on the contents of our configuration file.
            
            Here is the an example:
            ```bash
            # File Name: NLP/configs/tc/c20_agnews/agnews_c20_iid_noagg.yaml
            # Dataset
            dataset: agnews
            data_file_path: ../fednlp_data/data_files/agnews_data.h5
            partition_file_path: ../fednlp_data/partition_files/agnews_partition.h5
            partition_method: niid_label_clients=20_alpha=100
            reprocess_input_data: False
            num_labels: 4
            # Model Related
            model_type: bert
            model_class: ClassificationModel
            model: bert-base-uncased
            ```

            This line: `partition_method: niid_label_clients=20_alpha=100` shows that we need to partition the dataset of 20 clients with an iid degree of 100.

## 3. Reproduce our results

After a complex data processing session, we can finally run the code, congratulations!

- The entry point for all experiments is `simulation.py`

- run the following commands to run a experiment (all configs are stored in the `configs` folder):
    ```bash
    python simulation.py --cfg_path <YOUR_CONFIG_PATH>
    ```

- Due to possible bugs in the Flowers framework itself, we could not follow the instructions given on the [official website](https://flower.ai/docs/framework/example-pytorch-from-centralized-to-federated.html) for distributed training, and could only perform simulation experiments. For the details, pls refer to [this issue](https://github.com/adap/flower/issues/2167).

