import torch as t
import torch.nn as nn
from models.LiGO import LiGOLinear
import logging
from collections import OrderedDict
import torch.nn.functional as F
import os
from torch import Tensor
from logging import DEBUG, INFO, WARNING
from flwr.common.logger import log


# Define model
class LiGOMLP(nn.Module):
    def __init__(
        self,
        n_features=784,
        n_hiddens=100,
        n_layers=3,
        n_outputs=10,
        target_hiddens=None,
        target_layers=None,
        small_model=None,
    ):
        super(LiGOMLP, self).__init__()

        # save parameters to member variables
        self.n_features = n_features
        self.n_hiddens = n_hiddens
        self.n_layers = n_layers
        self.target_hiddens = target_hiddens
        self.target_layers = target_layers
        self.small_model = small_model
        # Construct the LiGO model
        self.net = OrderedDict(
            (
                ("flatten", nn.Flatten()),
                (
                    "feature",
                    LiGOLinear(
                        n_features,
                        n_hiddens,
                        target_out=target_hiddens,
                    ),
                ),
                (
                    "relu",
                    nn.ReLU(),
                ),
            )
        )
        for i in range(n_layers - 2):
            self.net["linear_{}".format(i)] = LiGOLinear(
                n_hiddens,
                n_hiddens,
                target_in=target_hiddens,
                target_out=target_hiddens,
            )
            self.net["relu_{}".format(i)] = nn.ReLU()
        self.net["classifier"] = LiGOLinear(
            n_hiddens,
            n_outputs,
            target_in=target_hiddens,
        )
        self.net = nn.Sequential(self.net)
        # Width Expansion Parameters
        if target_layers is not None:
            self.depth_expansion = nn.Parameter(
                t.ones((target_layers - 2, n_layers - 2)) / (n_layers - 2)
            )
            log(INFO, "Target layers is specified, enable depth_expansion module.")

        # warning if there are no small model's dict
        if (
            target_hiddens is not None or target_layers is not None
        ) and small_model is None:
            log(
                WARNING,
                "There is no pretrained small model, this model will use the default initializated small model to perform LiGO training!",
            )
        # load the small model, it can be either dict or the model path
        if small_model is not None:
            if isinstance(small_model, str):
                self.net.load_state_dict(t.load(small_model), False)
            elif isinstance(small_model, dict):
                self.net.load_state_dict(small_model, strict=False)

    @property
    def middle_layer_wandb(self):
        weights = t.cat(
            [
                getattr(self.net, "linear_{}".format(i)).merged_wandb[0].unsqueeze(0)
                for i in range(self.depth_expansion.shape[1])
            ]
        )
        biases = t.cat(
            [
                getattr(self.net, "linear_{}".format(i)).merged_wandb[1].unsqueeze(0)
                for i in range(self.depth_expansion.shape[1])
            ]
        )
        return weights, biases

    @property
    def expanded_model_state_dict(self):
        if self.target_hiddens is not None or self.target_layers is not None:
            with t.no_grad():
                state_dict = {}
                # save the head and tail parameters
                (
                    state_dict["net.feature.weight"],
                    state_dict["net.feature.bias"],
                ) = self.net.feature.merged_wandb

                (
                    state_dict["net.classifier.weight"],
                    state_dict["net.classifier.bias"],
                ) = self.net.classifier.merged_wandb
                # save the middle layer weight and bias
                for i in range(self.depth_expansion.shape[0]):
                    weight, bias = self.middle_layer_wandb
                    depth_weight = self.depth_expansion[i]
                    weight = (depth_weight.unsqueeze(-1).unsqueeze(-1) * weight).sum(
                        0, keepdim=False
                    )
                    bias = (depth_weight.unsqueeze(-1) * bias).sum(0, keepdim=False)

                    state_dict["net.linear_{}.weight".format(i)] = weight
                    state_dict["net.linear_{}.bias".format(i)] = bias

            return state_dict
        else:
            return None

    @property
    def ligo_dict(self):
        sd = self.state_dict()
        keys = list(sd.keys())
        for k in keys:
            if "LiGO" not in k:
                del sd[k]
        return sd

    def forward(self, x: t.Tensor):
        if hasattr(self, "depth_expansion"):
            x = self.net.flatten(x)
            x = self.net.feature(x)
            x = self.net.relu(x)

            for i in range(self.depth_expansion.shape[0]):
                weight, bias = self.middle_layer_wandb
                depth_weight = self.depth_expansion[i]
                weight = (depth_weight.unsqueeze(-1).unsqueeze(-1) * weight).sum(
                    0, keepdim=False
                )
                bias = (depth_weight.unsqueeze(-1) * bias).sum(0, keepdim=False)
                x = F.linear(x, weight, bias)
                x = F.relu(x)

            x = self.net.classifier(x)
            return x
        else:
            return self.net.forward(x)

    def save_model(self, model_dir: str):
        os.mkdir(model_dir)
        t.save(self.state_dict(), os.path.join(model_dir, "original_model.pth"))
        t.save(
            self.expanded_model_state_dict,
            os.path.join(model_dir, "enlarged_model.pth"),
        )


# test
if __name__ == "__main__":
    model = LiGOMLP(768, 100, 4, 10, 127, 5, None)
    model.forward(t.randn((1, 16, 16, 3)))
