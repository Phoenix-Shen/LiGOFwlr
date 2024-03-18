import yaml
from data.span_extraction.data_loader import load
from engine.seq_tagging_trainer import TrainerForSeqTagging
from engine.classification_trainer import TrainerForClassification
from engine.span_extraction_trainer import TrainerForSpanExtraction
from model_args import SeqTaggingArgs
from transformers import (
    BertConfig,
    BertForTokenClassification,
    BertForSequenceClassification,
    BertForQuestionAnswering,
)

if __name__ == "__main__":
    args = SeqTaggingArgs()

    with open("configs/span_extraction.yaml", "r") as f:
        arg_yaml = yaml.load(f, yaml.FullLoader)
    args.update_from_dict(arg_yaml)

    dataset, output_dim = load(args)

    bert_config = BertConfig()
    bert_config.num_labels = 2
    print(output_dim)
    model = BertForQuestionAnswering(bert_config)

    idx = 0
    train_data, test_data = dataset[5][idx], dataset[6][idx]
    trainer = TrainerForSpanExtraction()

    print(trainer.train(model, train_data, "cuda:1", args, test_data))
    print(trainer.test(model, test_data, "cuda:1", args))
