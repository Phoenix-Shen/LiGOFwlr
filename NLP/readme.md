1. Prepare the data
download the data from FedML official
2. Prepare the model ckpt
need to install tensorflow
`transformers-cli convert --model_type bert  --tf_checkpoint bert_model.ckpt --config bert_config.json  --pytorch_dump_output pytorch_model.bin`


3. Plan

- BertForSequenceClassification
    - 20news
    - agnews
- BertForQuestionAnswering
    - squad_1.1
    - mrqa
- BertForTokenClassification
    - wikiner
    - semeval_2010_task8

4. Install 

pip install torch torchvision torchaudio transformers evaluate datasets flwr[simulation] h5py wandb timm scikit-learn seqeval

5. prepare bert_tokenizer to ./pretrained_models/bert_tokenizer, it can be downloaded via huggingface. 