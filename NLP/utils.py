import torch
from transformers import (
    BertConfig,
    BertForTokenClassification,
    BertForQuestionAnswering,
    BertForSequenceClassification,
)
from models.BERT import LiGOBERT
import warnings
import torch.nn as nn
import torch.optim as optim
import numpy as np
from copy import deepcopy
from flwr.common import NDArrays, EvaluateRes, FitRes
from typing import Dict, Union, Optional
import wandb
from engine.classification_trainer import TrainerForClassification
from engine.seq_tagging_trainer import TrainerForSeqTagging
from engine.span_extraction_trainer import TrainerForSpanExtraction
import data.text_classification.data_loader
import data.seq_tagging.data_loader
import data.span_extraction.data_loader
from model_args import ModelArgs, SeqTaggingArgs, SpanExtractionArgs, ClassificationArgs

Scalar = Union[bool, bytes, float, int, str]

config_dict = {"bert": (BertConfig, LiGOBERT)}
model_dict = {
    "bert": {
        "QuestionAnsweringModel": BertForQuestionAnswering,
        "SeqTaggingModel": BertForTokenClassification,
        "ClassificationModel": BertForSequenceClassification,
    }
}
trainer_dict = {
    "bert": {
        "QuestionAnsweringModel": TrainerForSpanExtraction,
        "SeqTaggingModel": TrainerForSeqTagging,
        "ClassificationModel": TrainerForClassification,
    }
}
dataset_dict = {
    "QuestionAnsweringModel": data.span_extraction.data_loader.load,
    "SeqTaggingModel": data.seq_tagging.data_loader.load,
    "ClassificationModel": data.text_classification.data_loader.load,
}


def construct_model(
    model_type: str,
    model_class: str,
    kwargs: dict,
) -> nn.Module:
    """construct a model according to the given parameters

    Args:
        model_type (str): bert or gpt2
        model_class (str): what type of task do you want to perform?
        kwargs (dict): model kwargs for construct the model

    Returns:
        nn.Module: the desired model.
    """
    kwargs = {} if kwargs is None else kwargs
    config_class, ligo_class = config_dict[model_type]
    small_model_config = config_class(**kwargs["small_model"])
    if "large_model" in kwargs.keys() and kwargs["large_model"] is not None:
        large_model_config = config_class(**kwargs["large_model"])
    else:
        large_model_config = None
    model = ligo_class(
        small_model_config, large_model_config, model_dict[model_type][model_class]
    )
    return model


def load_args(args: dict) -> ModelArgs:
    args_dict = {
        "ClassificationModel": ClassificationArgs,
        "SeqTaggingModel": SeqTaggingArgs,
        "QuestionAnsweringModel": SpanExtractionArgs,
    }
    return args_dict[args["model_class"]]()


def construct_dataset(
    args: ModelArgs,
):
    dataset, class_num = dataset_dict[args.model_class](args)
    # dataset = [
    #     train_data_num,
    #     test_data_num,
    #     train_data_global,
    #     test_data_global,
    #     train_data_local_num_dict,
    #     train_data_local_dict,
    #     test_data_local_dict,
    #     class_num,
    # ]
    (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = dataset
    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )


def construct_trainer(
    model_type: str,
    model_class: str,
):
    return trainer_dict[model_type][model_class]()


def gen_hetro_model_args(args: dict) -> dict:
    """Generates heterogeneous model architectures according to arguments.

    Args:
        args (dict): the arguments.
    """
    n_type_models = len(args.model_kwargs.keys())
    model_type = np.random.choice(np.arange(n_type_models))
    return int(model_type)


def get_model_params(model: nn.Module, mode: int = 0) -> NDArrays:
    """Returns a model's parameters.

    Args:
        model (nn.Module): _description_
        mode (int, optional): _description_. Defaults to 0: state_dict(), 1:expanded_model_state_dict, 2:ligo_dict

    Returns:
        NDArrays: list of model's parameters (ndarray format)
    """
    if mode == 0:
        return [val.cpu().numpy() for _, val in model.state_dict().items()]
    elif mode == 1:
        return [val.cpu().numpy() for _, val in model.expanded_model_state_dict.items()]
    elif mode == 2:
        return [val.cpu().numpy() for _, val in model.ligo_dict.items()]
    else:
        raise ValueError("Invalid mode")


def set_seed(seed: int):
    """set seed for numpy and pytorch

    Args:
        seed (int): the random seed
    """
    torch.manual_seed(seed)  # set the seed for Pytorch
    np.random.seed(seed)  # set the seed for numpy
    torch.cuda.manual_seed_all(seed)  # Set cuda manual seed
    torch.backends.cudnn.deterministic = True  # Set deterministic = True


def weighted_metrics_avg(metrics: list[tuple[Dict[str, Scalar], int]]) -> dict:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum([num_examples for _, num_examples in metrics])
    # Initialize an empty dictionary to store the aggregated metrics
    aggregated_metrics = {}
    # Loop over the keys of the metrics dictionary
    if len(metrics) > 0:
        for key in metrics[0][0].keys():
            # Calculate the weighted average of the metric values from all clients
            weighted_sum = sum(
                [metric[key] * num_examples for metric, num_examples in metrics]
            )
            weighted_avg = weighted_sum / num_total_evaluation_examples
            # Store the weighted average value in the aggregated metrics dictionary
            aggregated_metrics[key] = weighted_avg
    # Return the aggregated metrics dictionary
    return aggregated_metrics


def log_wandb(results: Union[list[FitRes], list[EvaluateRes]]) -> None:
    """Log to wandb

    Args:
        results (Union[FitRes,EvaluateRes]): the result collected by the server
    """
    # for different types of Result, we have different manners to handle that.
    if isinstance(results[0], FitRes):
        tmp_dict = {}
        for idx, result in enumerate(results):
            for key in result.metrics.keys():
                tmp_dict["client#{}".format(idx) + key] = result.metrics[key]
        wandb.log(tmp_dict,commit=False)

    if isinstance(results[0], EvaluateRes):
        tmp_dict = {}
        for idx, result in enumerate(results):
            for key in result.metrics.keys():
                tmp_dict["client#{}".format(idx) + key] = result.metrics[key]
            tmp_dict["client#{}".format(idx) + "test_loss"] = result.loss
        wandb.log(tmp_dict)
