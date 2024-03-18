import torch as t
import torch.nn as nn
import math
import torch.nn.functional as F


class LiGOLinear(nn.Linear):
    """Implementation of LiGOLinear layer, which is similar to the LoRA operator."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        target_in: int = None,
        target_out: int = None,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        # Important to call super.__init__
        super().__init__(in_features, out_features, bias, device, dtype)
        # init weight value has shape [out,in]
        if self.in_features != target_in and target_in is not None:
            self.LiGOA = nn.Parameter(t.empty((self.in_features, target_in)))
            self.weight.requires_grad = False
            self.bias.requires_grad = False
        if self.out_features != target_out and target_out is not None:
            self.LiGOB = nn.Parameter(t.empty((self.out_features, target_out)))
            self.weight.requires_grad = False
            self.bias.requires_grad = False
        # call reset parameters function
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        if hasattr(self, "LiGOA"):
            nn.init.kaiming_uniform_(self.LiGOA, a=math.sqrt(5))
        if hasattr(self, "LiGOB"):
            nn.init.kaiming_uniform_(self.LiGOB, a=math.sqrt(5))

    def train(self, mode=True):
        if hasattr(self, "LiGOA") or hasattr(self, "LiGOB"):
            if mode:
                if hasattr(self, "LiGOA"):
                    self.LiGOA.requires_grad = mode
                if hasattr(self, "LiGOB"):
                    self.LiGOB.requires_grad = mode
        else:
            nn.Linear.train(self, mode)

    def forward(self, x: t.Tensor):
        weight = self.weight
        bias = self.bias

        if hasattr(self, "LiGOB"):
            weight = self.LiGOB.T @ weight
            bias = self.LiGOB.T @ bias

        if hasattr(self, "LiGOA"):
            weight = weight @ self.LiGOA
            bias = bias
        return F.linear(
            x,
            weight=weight,
            bias=bias,
        )

    @property
    def merged_wandb(self):
        weight = self.weight
        bias = self.bias

        if hasattr(self, "LiGOB"):
            weight = self.LiGOB.T @ weight
            bias = self.LiGOB.T @ bias

        if hasattr(self, "LiGOA"):
            weight = weight @ self.LiGOA
            bias = bias
        return weight, bias
