from torch import Tensor
from collections import OrderedDict
import torch as t
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, Block
from transformers.models.bert import (
    BertConfig,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertPreTrainedModel,
    BertForPreTraining,
    BertModel,
    BertForQuestionAnswering,
)
from transformers import PretrainedConfig
from typing import Optional, Union


class LiGOBERT(nn.Module):
    def __init__(
        self,
        small_model_config: BertConfig,
        large_model_config: Optional[BertConfig],
        model_type=BertPreTrainedModel,
    ):
        super().__init__()
        # 1. should we enable ligo model?
        self.model_type = model_type
        self.enable_ligo = large_model_config is not None
        # 2. construct small model & large model
        if hasattr(small_model_config, "pretrained_path"):
            small_model = model_type.from_pretrained(
                small_model_config.pretrained_path, config=small_model_config
            )
            print(
                f"load {small_model_config.pretrained_path} as pretrained small model."
            )
        else:
            small_model = model_type(small_model_config)

        # 3. if the ligo operator is enabled, we initialize the ligo operators
        if self.enable_ligo:
            # 3.1 construct large model according to the config
            large_model = model_type(large_model_config)
            # 3.2 initialize member variables
            self.small_model = small_model.requires_grad_(False)
            self.large_model = large_model.requires_grad_(True)

            self.D1 = self.small_model.config.hidden_size
            self.D2 = self.large_model.config.hidden_size
            self.L1 = self.small_model.config.num_hidden_layers
            self.L2 = self.large_model.config.num_hidden_layers
            # 3.3 initialize embedding expansion operator
            self.embedding_expansion = nn.Parameter(t.empty(size=(self.D2, self.D1)))
            # 3.4 initialize width expansion operator (B_l W_l A_l^T)
            self.width_expansion_operator = nn.ParameterDict()
            for l in range(self.L1):
                # Attention QKVs
                self.width_expansion_operator["Weight_B_Q_{}".format(l)] = nn.Parameter(
                    t.empty(size=(self.D2, self.D1))
                )

                self.width_expansion_operator["Weight_B_K_{}".format(l)] = nn.Parameter(
                    t.empty(size=(self.D2, self.D1))
                )

                self.width_expansion_operator["Weight_B_V_{}".format(l)] = nn.Parameter(
                    t.empty(size=(self.D2, self.D1))
                )

                # Feed-Forward NN
                self.width_expansion_operator[
                    "Weight_B_fc1_{}".format(l)
                ] = nn.Parameter(t.empty(size=(self.D2, self.D1)))
            # pooling layer in huggingface BERT models
            self.width_expansion_operator["Weight_B_pooling"] = nn.Parameter(
                t.empty(size=(self.D2, self.D1))
            )
            # 3.5 initialize depth expansion operator (w_ij)
            # 8 denotes [q ,k ,v ,o ,ln1 ,fc1 ,fc2 ,ln2]
            self.depth_expansion_operator = nn.Parameter(
                t.empty(size=(8, self.L2, self.L1))
            )
            # 3.6 initialize the ligo operators via xavier normal
            self._reset_ligo_parameters()
        # 4. else we only keep the small model
        else:
            self.small_model = small_model.requires_grad_(True)

    def _reset_ligo_parameters(self):
        nn.init.xavier_normal_(self.embedding_expansion)

        for key in self.width_expansion_operator.keys():
            nn.init.xavier_normal_(self.width_expansion_operator[key])

        self.depth_expansion_operator.data.fill_(1/self.L1)

    def expand_params(self):
        """Expands the small_model to the large_model via ligo operator

        Returns:
            None
        """
        self.full_dict = {}
        # 1. Embedding Expansion
        # expand parameters
        word_embed_wide = (
            self.small_model.bert.embeddings.word_embeddings.weight
            @ self.embedding_expansion.T
        )
        pos_embed_wide = (
            self.small_model.bert.embeddings.position_embeddings.weight
            @ self.embedding_expansion.T
        )
        token_type_embed_wide = (
            self.small_model.bert.embeddings.token_type_embeddings.weight
            @ self.embedding_expansion.T
        )
        norm_weight_wide = (
            self.embedding_expansion @ self.small_model.bert.embeddings.LayerNorm.weight
        )
        norm_bias_wide = (
            self.embedding_expansion @ self.small_model.bert.embeddings.LayerNorm.bias
        )
        # copy parameters
        self.large_model.bert.embeddings.word_embeddings.weight.data.copy_(
            word_embed_wide
        )
        self.large_model.bert.embeddings.position_embeddings.weight.data.copy_(
            pos_embed_wide
        )
        self.large_model.bert.embeddings.token_type_embeddings.weight.data.copy_(
            token_type_embed_wide
        )
        self.large_model.bert.embeddings.LayerNorm.weight.data.copy_(norm_weight_wide)
        self.large_model.bert.embeddings.LayerNorm.bias.data.copy_(norm_bias_wide)
        # save to dict
        self.full_dict["bert.embeddings.word_embeddings.weight"] = word_embed_wide
        self.full_dict["bert.embeddings.position_embeddings.weight"] = pos_embed_wide
        self.full_dict[
            "bert.embeddings.token_type_embeddings.weight"
        ] = token_type_embed_wide
        self.full_dict["bert.embeddings.LayerNorm.weight"] = norm_weight_wide
        self.full_dict["bert.embeddings.LayerNorm.bias"] = norm_bias_wide
        # 2. Expand width for each layer (block)
        params: OrderedDict[str, Tensor] = OrderedDict()
        for idx, (name, layer) in enumerate(
            self.small_model.bert.encoder.layer.named_children()
        ):
            params[name + "_wide"] = {}
            wider_layer = params[name + "_wide"]
            # 2.1 self attention
            W_Q: Tensor = layer.attention.self.query.weight
            W_K: Tensor = layer.attention.self.key.weight
            W_V: Tensor = layer.attention.self.value.weight
            W_O: Tensor = layer.attention.output.dense.weight

            b_Q: Tensor = layer.attention.self.query.bias
            b_K: Tensor = layer.attention.self.key.bias
            b_V: Tensor = layer.attention.self.value.bias
            b_O: Tensor = layer.attention.output.dense.bias

            wider_layer["attention.self.query.weight"] = (
                self.width_expansion_operator["Weight_B_Q_{}".format(idx)]
                @ W_Q
                @ self.embedding_expansion.T
            )
            wider_layer["attention.self.key.weight"] = (
                self.width_expansion_operator["Weight_B_K_{}".format(idx)]
                @ W_K
                @ self.embedding_expansion.T
            )
            wider_layer["attention.self.value.weight"] = (
                self.width_expansion_operator["Weight_B_V_{}".format(idx)]
                @ W_V
                @ self.embedding_expansion.T
            )
            wider_layer["attention.output.dense.weight"] = (
                self.embedding_expansion
                @ W_O
                @ self.width_expansion_operator["Weight_B_V_{}".format(idx)].T
            )

            wider_layer["attention.self.query.bias"] = (
                self.width_expansion_operator["Weight_B_Q_{}".format(idx)] @ b_Q
            )
            wider_layer["attention.self.key.bias"] = (
                self.width_expansion_operator["Weight_B_K_{}".format(idx)] @ b_K
            )
            wider_layer["attention.self.value.bias"] = (
                self.width_expansion_operator["Weight_B_V_{}".format(idx)] @ b_V
            )
            wider_layer["attention.output.dense.bias"] = self.embedding_expansion @ b_O

            # 2.2 layer norm
            wider_layer["attention.output.LayerNorm.weight"] = (
                self.embedding_expansion @ layer.attention.output.LayerNorm.weight
            )
            wider_layer["attention.output.LayerNorm.bias"] = (
                self.embedding_expansion @ layer.attention.output.LayerNorm.bias
            )

            wider_layer["output.LayerNorm.weight"] = (
                self.embedding_expansion @ layer.output.LayerNorm.weight
            )
            wider_layer["output.LayerNorm.bias"] = (
                self.embedding_expansion @ layer.output.LayerNorm.bias
            )

            # 2.3 feed forward nn
            wider_layer["intermediate.dense.weight"] = (
                self.width_expansion_operator["Weight_B_fc1_{}".format(idx)]
                @ W_V
                @ self.embedding_expansion.T
            )
            wider_layer["intermediate.dense.bias"] = (
                self.width_expansion_operator["Weight_B_fc1_{}".format(idx)] @ b_V
            )

            wider_layer["output.dense.weight"] = (
                self.embedding_expansion
                @ W_V
                @ self.width_expansion_operator["Weight_B_fc1_{}".format(idx)].T
            )
            wider_layer["output.dense.bias"] = (
                self.embedding_expansion @ layer.output.dense.bias
            )

        # 3. Expand depth

        for idx_large, (name, layer) in enumerate(
            self.large_model.bert.encoder.layer.named_children()
        ):
            # 3.1 allocate a dictionary
            state_dict = {
                "attention.self.query.weight": t.zeros_like(
                    layer.attention.self.query.weight
                )
                .unsqueeze(0)
                .repeat(self.L1, 1, 1),
                "attention.self.key.weight": t.zeros_like(
                    layer.attention.self.key.weight
                )
                .unsqueeze(0)
                .repeat(self.L1, 1, 1),
                "attention.self.value.weight": t.zeros_like(
                    layer.attention.self.value.weight
                )
                .unsqueeze(0)
                .repeat(self.L1, 1, 1),
                "attention.output.dense.weight": t.zeros_like(
                    layer.attention.output.dense.weight
                )
                .unsqueeze(0)
                .repeat(self.L1, 1, 1),
                "attention.self.query.bias": t.zeros_like(
                    layer.attention.self.query.bias
                )
                .unsqueeze(0)
                .repeat(self.L1, 1),
                "attention.self.key.bias": t.zeros_like(layer.attention.self.key.bias)
                .unsqueeze(0)
                .repeat(self.L1, 1),
                "attention.self.value.bias": t.zeros_like(
                    layer.attention.self.value.bias
                )
                .unsqueeze(0)
                .repeat(self.L1, 1),
                "attention.output.dense.bias": t.zeros_like(
                    layer.attention.output.dense.bias
                )
                .unsqueeze(0)
                .repeat(self.L1, 1),
                "attention.output.LayerNorm.weight": t.zeros_like(
                    layer.attention.output.LayerNorm.weight
                )
                .unsqueeze(0)
                .repeat(self.L1, 1),
                "attention.output.LayerNorm.bias": t.zeros_like(
                    layer.attention.output.LayerNorm.bias
                )
                .unsqueeze(0)
                .repeat(self.L1, 1),
                "output.LayerNorm.weight": t.zeros_like(layer.output.LayerNorm.weight)
                .unsqueeze(0)
                .repeat(self.L1, 1),
                "output.LayerNorm.bias": t.zeros_like(layer.output.LayerNorm.bias)
                .unsqueeze(0)
                .repeat(self.L1, 1),
                "intermediate.dense.weight": t.zeros_like(
                    layer.intermediate.dense.weight
                )
                .unsqueeze(0)
                .repeat(self.L1, 1, 1),
                "intermediate.dense.bias": t.zeros_like(layer.intermediate.dense.bias)
                .unsqueeze(0)
                .repeat(self.L1, 1),
                "output.dense.weight": t.zeros_like(layer.output.dense.weight)
                .unsqueeze(0)
                .repeat(self.L1, 1, 1),
                "output.dense.bias": t.zeros_like(layer.output.dense.bias)
                .unsqueeze(0)
                .repeat(self.L1, 1),
            }
            # 3.2 for each layer of the large model, we compute the weight sum of the prevous enlarged layer
            for idx_small, layer_key in enumerate(params.keys()):
                state_dict["attention.self.query.weight"][idx_small] = (
                    params[layer_key]["attention.self.query.weight"]
                    * self.depth_expansion_operator[0, idx_large, idx_small]
                )
                state_dict["attention.self.key.weight"][idx_small] = (
                    params[layer_key]["attention.self.key.weight"]
                    * self.depth_expansion_operator[1, idx_large, idx_small]
                )
                state_dict["attention.self.value.weight"][idx_small] = (
                    params[layer_key]["attention.self.value.weight"]
                    * self.depth_expansion_operator[2, idx_large, idx_small]
                )
                state_dict["attention.output.dense.weight"][idx_small] = (
                    params[layer_key]["attention.output.dense.weight"]
                    * self.depth_expansion_operator[3, idx_large, idx_small]
                )

                state_dict["attention.self.query.bias"][idx_small] = (
                    params[layer_key]["attention.self.query.bias"]
                    * self.depth_expansion_operator[0, idx_large, idx_small]
                )
                state_dict["attention.self.key.bias"][idx_small] = (
                    params[layer_key]["attention.self.key.bias"]
                    * self.depth_expansion_operator[1, idx_large, idx_small]
                )
                state_dict["attention.self.value.bias"][idx_small] = (
                    params[layer_key]["attention.self.value.bias"]
                    * self.depth_expansion_operator[2, idx_large, idx_small]
                )
                state_dict["attention.output.dense.bias"][idx_small] = (
                    params[layer_key]["attention.output.dense.bias"]
                    * self.depth_expansion_operator[3, idx_large, idx_small]
                )

                state_dict["attention.output.LayerNorm.weight"][idx_small] = (
                    params[layer_key]["attention.output.LayerNorm.weight"]
                    * self.depth_expansion_operator[4, idx_large, idx_small]
                )
                state_dict["attention.output.LayerNorm.bias"][idx_small] = (
                    params[layer_key]["attention.output.LayerNorm.bias"]
                    * self.depth_expansion_operator[4, idx_large, idx_small]
                )

                state_dict["output.LayerNorm.weight"][idx_small] = (
                    params[layer_key]["output.LayerNorm.weight"]
                    * self.depth_expansion_operator[5, idx_large, idx_small]
                )
                state_dict["output.LayerNorm.bias"][idx_small] = (
                    params[layer_key]["output.LayerNorm.bias"]
                    * self.depth_expansion_operator[5, idx_large, idx_small]
                )

                state_dict["intermediate.dense.weight"][idx_small] = (
                    params[layer_key]["intermediate.dense.weight"]
                    * self.depth_expansion_operator[6, idx_large, idx_small]
                )
                state_dict["intermediate.dense.bias"][idx_small] = (
                    params[layer_key]["intermediate.dense.bias"]
                    * self.depth_expansion_operator[6, idx_large, idx_small]
                )

                state_dict["output.dense.weight"][idx_small] = (
                    params[layer_key]["output.dense.weight"]
                    * self.depth_expansion_operator[7, idx_large, idx_small]
                )
                state_dict["output.dense.bias"][idx_small] = (
                    params[layer_key]["output.dense.bias"]
                    * self.depth_expansion_operator[7, idx_large, idx_small]
                )
            # 3.3 sum the parameters
            for key in state_dict.keys():
                state_dict[key] = t.sum(state_dict[key], dim=0)
            # 3.4 assign parameters to the layer
            layer.attention.self.query.weight.data.copy_(
                state_dict["attention.self.query.weight"]
            )
            layer.attention.self.key.weight.data.copy_(
                state_dict["attention.self.key.weight"]
            )
            layer.attention.self.value.weight.data.copy_(
                state_dict["attention.self.value.weight"]
            )
            layer.attention.output.dense.weight.data.copy_(
                state_dict["attention.output.dense.weight"]
            )
            layer.attention.self.query.bias.data.copy_(
                state_dict["attention.self.query.bias"]
            )
            layer.attention.self.key.bias.data.copy_(
                state_dict["attention.self.key.bias"]
            )
            layer.attention.self.value.bias.data.copy_(
                state_dict["attention.self.value.bias"]
            )
            layer.attention.output.dense.bias.data.copy_(
                state_dict["attention.output.dense.bias"]
            )
            layer.attention.output.LayerNorm.weight.data.copy_(
                state_dict["attention.output.LayerNorm.weight"]
            )
            layer.attention.output.LayerNorm.bias.data.copy_(
                state_dict["attention.output.LayerNorm.bias"]
            )
            layer.attention.output.dense.weight.data.copy_(
                state_dict["attention.output.dense.weight"]
            )
            layer.attention.output.dense.bias.data.copy_(
                state_dict["attention.output.dense.bias"]
            )
            layer.intermediate.dense.weight.data.copy_(
                state_dict["intermediate.dense.weight"]
            )
            layer.intermediate.dense.bias.data.copy_(
                state_dict["intermediate.dense.bias"]
            )
            layer.output.dense.weight.data.copy_(state_dict["output.dense.weight"])
            layer.output.dense.bias.data.copy_(state_dict["output.dense.bias"])

            # 3.5 store the enlarged parameters
            self.full_dict["bert.encoder.layer." + name] = state_dict
        # 4. Other layers such as classifier & qa head
        if self.model_type == BertForSequenceClassification:
            self.full_dict["bert.pooler.dense.weight"] = (
                self.width_expansion_operator["Weight_B_pooling"]
                @ self.small_model.bert.pooler.dense.weight
                @ self.embedding_expansion.T
            )
            self.large_model.bert.pooler.dense.weight.data.copy_(
                self.full_dict["bert.pooler.dense.weight"]
            )

            self.full_dict["bert.pooler.dense.bias"] = (
                self.width_expansion_operator["Weight_B_pooling"]
                @ self.small_model.bert.pooler.dense.bias
            )
            self.large_model.bert.pooler.dense.bias.data.copy_(
                self.full_dict["bert.pooler.dense.bias"]
            )

            self.full_dict["classifier.weight"] = (
                self.small_model.classifier.weight @ self.embedding_expansion.T
            )
            self.large_model.classifier.weight.data.copy_(
                self.full_dict["classifier.weight"]
            )
            self.full_dict["classifier.bias"] = self.small_model.classifier.bias
            self.large_model.classifier.bias.data.copy_(
                self.full_dict["classifier.bias"]
            )
        elif self.model_type == BertForTokenClassification:
            self.full_dict["classifier.weight"] = (
                self.small_model.classifier.weight @ self.embedding_expansion.T
            )
            self.large_model.classifier.weight.data.copy_(
                self.full_dict["classifier.weight"]
            )
            self.full_dict["classifier.bias"] = self.small_model.classifier.bias
            self.large_model.classifier.bias.data.copy_(
                self.full_dict["classifier.bias"]
            )
        elif self.model_type == BertForQuestionAnswering:
            self.full_dict["qa_outputs.weight"] = (
                self.small_model.qa_outputs.weight @ self.embedding_expansion.T
            )
            self.large_model.qa_outputs.weight.data.copy_(
                self.full_dict["qa_outputs.weight"]
            )
            self.full_dict["qa_outputs.bias"] = self.small_model.qa_outputs.bias
            self.large_model.qa_outputs.bias.data.copy_(
                self.full_dict["qa_outputs.bias"]
            )
        # if self.model_type == BertForPreTraining:
        #     self.full_dict["cls.predictions.bias"] = self.small_model.cls.predictions.bias
        #     self.full_dict["cls.predictions.weight"] = self.small_model.cls.predictions.weight
        #     self.full_dict[
        #         "cls.predictions.transform.dense.weight"
        #     ] = self.small_model.cls.predictions.transform.dense.weight
        #     self.full_dict[
        #         "cls.predictions.transform.dense.bias"
        #     ] = self.small_model.cls.predictions.transform.dense.bias
        #     self.full_dict[
        #         "cls.predictions.transform.LayerNorm.weight"
        #     ] = self.small_model.cls.predictions.transform.LayerNorm.weight
        #     self.full_dict[
        #         "cls.predictions.transform.LayerNorm.bias"
        #     ] = self.small_model.cls.predictions.transform.LayerNorm.bias
        #     self.full_dict[
        #         "cls.seq_relationship.weight"
        #     ] = self.small_model.cls.seq_relationship.weight
        #     self.full_dict[
        #         "cls.seq_relationship.bias"
        #     ] = self.small_model.cls.seq_relationship.bias

    def forward(self, *inputs, **kwargs):
        if self.enable_ligo:
            self.expand_params()
            out = self.large_model.forward(*inputs, **kwargs)
        else:
            out = self.small_model.forward(*inputs, **kwargs)
        return out

    @property
    def expanded_model_state_dict(self):
        self.expand_params()
        keys = self.large_model.state_dict().keys()
        sd = self.large_model.state_dict()
        for key in keys:
            sd["small_model." + key] = sd.pop(key)
        return sd

    @property
    def ligo_dict(self):
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            if "small_model" in key or "large_model" in key:
                del state_dict[key]
        return state_dict

    def zero_grad_largemodel(self):
        for name, param in self.large_model.named_parameters():
            if param.grad is not None:
                param.grad.zero_()

    def backward_param(self):
        if hasattr(self, "large_model"):
            for name, param in self.large_model.named_parameters():
                if "layer" in name:
                    out_layer_key = ".".join(name.split(".")[:4])
                    inner_layer_key = ".".join(name.split(".")[4:])
                    self.full_dict[out_layer_key][inner_layer_key].backward(
                        param.grad, retain_graph=True
                    )
                else:
                    if self.full_dict[name].requires_grad:
                        self.full_dict[name].backward(param.grad, retain_graph=True)
        else:
            pass

    def train(self, mode: bool = True):
        if self.enable_ligo:
            if not isinstance(mode, bool):
                raise ValueError("training mode is expected to be boolean")
            self.training = mode
            for module in self.children():
                module.train(mode)
            self.small_model = self.small_model.requires_grad_(False)
            self.small_model.train(False)
        else:
            self.small_model.train(mode)
        return self

    def eval(self):
        return self.train(False)
