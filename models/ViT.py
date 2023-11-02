from torch import Tensor
from collections import OrderedDict
import torch as t
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, Block


class LiGOViT(nn.Module):
    def __init__(
        self,
        patch_size=16,
        n_hiddens=192,
        n_layers=12,
        num_heads=3,
        target_hiddens=None,
        target_layers=None,
        target_heads=None,
        small_model_path=None,
        num_classes=100,
    ) -> None:
        super().__init__()
        # has ligo operator
        self.enable_ligo = (
            target_hiddens is not None
            and target_layers is not None
            and target_heads is not None
        )
        # construct visiontransformers according to the kwargs.
        small_model = VisionTransformer(
            patch_size=patch_size,
            embed_dim=n_hiddens,
            depth=n_layers,
            num_heads=num_heads,
            num_classes=num_classes,
            mlp_ratio=1,
        )
        # Load pretrained small models.
        if small_model_path is not None:
            small_model.load_state_dict(t.load(small_model_path)["model"])
        # if there are ligo operators, inititalize the ligo operators
        if self.enable_ligo:
            large_model = VisionTransformer(
                patch_size=patch_size,
                embed_dim=target_hiddens,
                depth=target_layers,
                num_heads=target_heads,
                num_classes=num_classes,
                mlp_ratio=1,
            )

            # initialize member variables
            self.small_model = small_model.requires_grad_(False)
            self.large_model = large_model.requires_grad_(True)
            self.D1 = small_model.embed_dim
            self.D2 = large_model.embed_dim
            self.L1 = len(list(small_model.blocks.children()))
            self.L2 = len(list(large_model.blocks.children()))
            ###############################################
            # Define the parameters for the LiGO operator #
            ###############################################
            # Embedding Expansion Operator
            self.embedding_expansion = nn.Parameter(t.empty(size=(self.D2, self.D1)))

            # Width Expansion Operator (B_l W_l A_l^T)
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

            # Depth Expansion Operator w_ij
            # 8 denotes q k v o ln1 fc1 fc2 ln2
            self.depth_expansion_operator = nn.Parameter(
                t.empty(size=(8, self.L2, self.L1))
            )

            self._reset_parameters()
        else:
            self.small_model = small_model.requires_grad_(True)

    def expand_params(
        self,
    ):
        ###############################################################
        # 1. Embedding Expansion
        # ViT does not have word embeddings, so skip this step

        # 2. Expand width for each Encoder Layer
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for idx, (name, module) in enumerate(self.small_model.blocks.named_children()):
            # Get the small Encoder Layer
            layer: Block = module

            # Construct a larger model
            layers[name + "_wide"] = {
                "attn.qkv.weight": t.zeros(
                    (self.D2 * 3, self.D2), device=layer.attn.qkv.weight.device
                ),
                "attn.qkv.bias": t.zeros(
                    (self.D2 * 3), device=layer.attn.qkv.weight.device
                ),
            }
            wider_layer = layers[name + "_wide"]

            # Initialize the parameters
            # 1. QKV weight & bias
            W_Q = layer.attn.qkv.weight[0 * self.D1 : 1 * self.D1, :]
            W_K = layer.attn.qkv.weight[1 * self.D1 : 2 * self.D1, :]
            W_V = layer.attn.qkv.weight[2 * self.D1 : 3 * self.D1, :]

            W_O = layer.attn.proj.weight

            b_Q = layer.attn.qkv.bias[0 * self.D1 : 1 * self.D1]
            b_K = layer.attn.qkv.bias[1 * self.D1 : 2 * self.D1]
            b_V = layer.attn.qkv.bias[2 * self.D1 : 3 * self.D1]

            b_O = layer.attn.proj.bias

            wider_layer["attn.qkv.weight"][0 * self.D2 : 1 * self.D2, :] = (
                self.width_expansion_operator["Weight_B_Q_{}".format(idx)]
                @ W_Q
                @ self.embedding_expansion.T
            )
            wider_layer["attn.qkv.weight"][1 * self.D2 : 2 * self.D2, :] = (
                self.width_expansion_operator["Weight_B_K_{}".format(idx)]
                @ W_K
                @ self.embedding_expansion.T
            )
            wider_layer["attn.qkv.weight"][2 * self.D2 : 3 * self.D2, :] = (
                self.width_expansion_operator["Weight_B_V_{}".format(idx)]
                @ W_V
                @ self.embedding_expansion.T
            )
            wider_layer["attn.proj.weight"] = (
                self.embedding_expansion
                @ W_O
                @ self.width_expansion_operator["Weight_B_V_{}".format(idx)].T
            )

            wider_layer["attn.qkv.bias"][0 * self.D2 : 1 * self.D2] = (
                self.width_expansion_operator["Weight_B_Q_{}".format(idx)] @ b_Q
            )
            wider_layer["attn.qkv.bias"][1 * self.D2 : 2 * self.D2] = (
                self.width_expansion_operator["Weight_B_K_{}".format(idx)] @ b_K
            )
            wider_layer["attn.qkv.bias"][2 * self.D2 : 3 * self.D2] = (
                self.width_expansion_operator["Weight_B_V_{}".format(idx)] @ b_V
            )
            wider_layer["attn.proj.bias"] = self.embedding_expansion @ b_O

            # 2. norm layers' weight & bias
            wider_layer["norm1.weight"] = self.embedding_expansion @ layer.norm1.weight
            wider_layer["norm1.bias"] = self.embedding_expansion @ layer.norm1.bias

            wider_layer["norm2.weight"] = self.embedding_expansion @ layer.norm2.weight
            wider_layer["norm2.bias"] = self.embedding_expansion @ layer.norm2.bias

            # 3. mlp's nn.linear module, also including weight & bias
            wider_layer["mlp.fc1.weight"] = (
                self.width_expansion_operator["Weight_B_fc1_{}".format(idx)]
                @ layer.mlp.fc1.weight
                @ self.embedding_expansion.T
            )
            wider_layer["mlp.fc1.bias"] = (
                self.width_expansion_operator["Weight_B_fc1_{}".format(idx)]
                @ layer.mlp.fc1.bias
            )

            wider_layer["mlp.fc2.weight"] = (
                self.embedding_expansion
                @ layer.mlp.fc2.weight
                @ self.width_expansion_operator["Weight_B_fc1_{}".format(idx)].T
            )
            wider_layer["mlp.fc2.bias"] = self.embedding_expansion @ layer.mlp.fc2.bias

        # 3. Expand Width
        self.full_dict = {}
        for idx_large, (name, module) in enumerate(
            self.large_model.blocks.named_children()
        ):
            # Get the large Encoder Layer
            layer: Block = module
            # Weight average
            # it is hard to use iteration to handle this,
            # so I must hard code this.

            # layer.mlp.fc1 = nn.Linear(self.D2, self.D2).to(layer.attn.qkv.weight.device)
            # layer.mlp.fc2 = nn.Linear(self.D2, self.D2).to(layer.attn.qkv.weight.device)

            state_dict = {
                "attn.qkv.weight": t.zeros_like(layer.attn.qkv.weight)
                .unsqueeze(0)
                .repeat(self.L1, 1, 1),
                "attn.qkv.bias": t.zeros_like(layer.attn.qkv.bias)
                .unsqueeze(0)
                .repeat(self.L1, 1),
                "attn.proj.weight": t.zeros_like(layer.attn.proj.weight)
                .unsqueeze(0)
                .repeat(self.L1, 1, 1),
                "attn.proj.bias": t.zeros_like(layer.attn.proj.bias)
                .unsqueeze(0)
                .repeat(self.L1, 1),
                "norm1.weight": t.zeros_like(layer.norm1.weight)
                .unsqueeze(0)
                .repeat(self.L1, 1),
                "norm1.bias": t.zeros_like(layer.norm1.bias)
                .unsqueeze(0)
                .repeat(self.L1, 1),
                "norm2.weight": t.zeros_like(layer.norm2.weight)
                .unsqueeze(0)
                .repeat(self.L1, 1),
                "norm2.bias": t.zeros_like(layer.norm2.bias)
                .unsqueeze(0)
                .repeat(self.L1, 1),
                "mlp.fc1.weight": t.zeros_like(layer.mlp.fc1.weight)
                .unsqueeze(0)
                .repeat(self.L1, 1, 1),
                "mlp.fc1.bias": t.zeros_like(layer.mlp.fc1.bias)
                .unsqueeze(0)
                .repeat(self.L1, 1),
                "mlp.fc2.weight": t.zeros_like(layer.mlp.fc2.weight)
                .unsqueeze(0)
                .repeat(self.L1, 1, 1),
                "mlp.fc2.bias": t.zeros_like(layer.mlp.fc2.bias)
                .unsqueeze(0)
                .repeat(self.L1, 1),
            }
            for idx_small, layer_key in enumerate(layers.keys()):
                W_Q_w = layers[layer_key]["attn.qkv.weight"][
                    0 * self.D2 : 1 * self.D2, :
                ]
                W_K_w = layers[layer_key]["attn.qkv.weight"][
                    1 * self.D2 : 2 * self.D2, :
                ]
                W_V_w = layers[layer_key]["attn.qkv.weight"][
                    2 * self.D2 : 3 * self.D2, :
                ]
                W_O_w = layers[layer_key]["attn.proj.weight"]

                b_Q_w = layers[layer_key]["attn.qkv.bias"][0 * self.D2 : 1 * self.D2]
                b_K_w = layers[layer_key]["attn.qkv.bias"][1 * self.D2 : 2 * self.D2]
                b_V_w = layers[layer_key]["attn.qkv.bias"][2 * self.D2 : 3 * self.D2]
                b_O_w = layers[layer_key]["attn.proj.bias"]

                state_dict["attn.qkv.weight"][idx_small][
                    0 * self.D2 : 1 * self.D2, :
                ] = (W_Q_w * self.depth_expansion_operator[0, idx_large, idx_small])
                state_dict["attn.qkv.weight"][idx_small][
                    1 * self.D2 : 2 * self.D2, :
                ] = (W_K_w * self.depth_expansion_operator[1, idx_large, idx_small])
                state_dict["attn.qkv.weight"][idx_small][
                    2 * self.D2 : 3 * self.D2, :
                ] = (W_V_w * self.depth_expansion_operator[2, idx_large, idx_small])

                state_dict["attn.qkv.bias"][idx_small][0 * self.D2 : 1 * self.D2] = (
                    b_Q_w * self.depth_expansion_operator[0, idx_large, idx_small]
                )
                state_dict["attn.qkv.bias"][idx_small][1 * self.D2 : 2 * self.D2] = (
                    b_K_w * self.depth_expansion_operator[1, idx_large, idx_small]
                )
                state_dict["attn.qkv.bias"][idx_small][2 * self.D2 : 3 * self.D2] = (
                    b_V_w * self.depth_expansion_operator[2, idx_large, idx_small]
                )

                state_dict["attn.proj.weight"][idx_small] = (
                    W_O_w * self.depth_expansion_operator[3, idx_large, idx_small]
                )
                state_dict["attn.proj.bias"][idx_small] = (
                    b_O_w * self.depth_expansion_operator[3, idx_large, idx_small]
                )

                state_dict["norm1.weight"][idx_small] = (
                    layers[layer_key]["norm1.weight"]
                    * self.depth_expansion_operator[4, idx_large, idx_small]
                )
                state_dict["norm1.bias"][idx_small] = (
                    layers[layer_key]["norm1.bias"]
                    * self.depth_expansion_operator[4, idx_large, idx_small]
                )

                state_dict["mlp.fc1.weight"][idx_small] = (
                    layers[layer_key]["mlp.fc1.weight"]
                    * self.depth_expansion_operator[5, idx_large, idx_small]
                )
                state_dict["mlp.fc1.bias"][idx_small] = (
                    layers[layer_key]["mlp.fc1.bias"]
                    * self.depth_expansion_operator[5, idx_large, idx_small]
                )

                state_dict["mlp.fc2.weight"][idx_small] = (
                    layers[layer_key]["mlp.fc2.weight"]
                    * self.depth_expansion_operator[6, idx_large, idx_small]
                )
                state_dict["mlp.fc2.bias"][idx_small] = (
                    layers[layer_key]["mlp.fc2.bias"]
                    * self.depth_expansion_operator[6, idx_large, idx_small]
                )

                state_dict["norm2.weight"][idx_small] = (
                    layers[layer_key]["norm2.weight"]
                    * self.depth_expansion_operator[7, idx_large, idx_small]
                )
                state_dict["norm2.bias"][idx_small] = (
                    layers[layer_key]["norm2.bias"]
                    * self.depth_expansion_operator[7, idx_large, idx_small]
                )

            state_dict["attn.qkv.weight"] = t.sum(state_dict["attn.qkv.weight"], dim=0)
            state_dict["attn.qkv.bias"] = t.sum(state_dict["attn.qkv.bias"], dim=0)
            state_dict["attn.proj.weight"] = t.sum(
                state_dict["attn.proj.weight"], dim=0
            )
            state_dict["attn.proj.bias"] = t.sum(state_dict["attn.proj.bias"], dim=0)
            state_dict["norm1.weight"] = t.sum(state_dict["norm1.weight"], dim=0)
            state_dict["norm1.bias"] = t.sum(state_dict["norm1.bias"], dim=0)
            state_dict["norm2.weight"] = t.sum(state_dict["norm2.weight"], dim=0)
            state_dict["norm2.bias"] = t.sum(state_dict["norm2.bias"], dim=0)
            state_dict["mlp.fc1.weight"] = t.sum(state_dict["mlp.fc1.weight"], dim=0)
            state_dict["mlp.fc1.bias"] = t.sum(state_dict["mlp.fc1.bias"], dim=0)
            state_dict["mlp.fc2.weight"] = t.sum(state_dict["mlp.fc2.weight"], dim=0)
            state_dict["mlp.fc2.bias"] = t.sum(state_dict["mlp.fc2.bias"], dim=0)

            # assign parameters to layer

            layer.attn.qkv.weight.data.copy_(state_dict["attn.qkv.weight"])
            layer.attn.qkv.bias.data.copy_(state_dict["attn.qkv.bias"])

            layer.attn.proj.weight.data.copy_(state_dict["attn.proj.weight"])
            layer.attn.proj.bias.data.copy_(state_dict["attn.proj.bias"])

            layer.norm1.weight.data.copy_(state_dict["norm1.weight"])
            layer.norm1.bias.data.copy_(state_dict["norm1.bias"])

            layer.norm2.weight.data.copy_(state_dict["norm2.weight"])
            layer.norm2.bias.data.copy_(state_dict["norm2.bias"])

            layer.mlp.fc1.weight.data.copy_(state_dict["mlp.fc1.weight"])
            layer.mlp.fc1.bias.data.copy_(state_dict["mlp.fc1.bias"])

            layer.mlp.fc2.weight.data.copy_(state_dict["mlp.fc2.weight"])
            layer.mlp.fc2.bias.data.copy_(state_dict["mlp.fc2.bias"])

            self.full_dict["blocks." + name] = state_dict

        # classification head
        self.full_dict[
            "patch_embed.proj.weight"
        ] = self.embedding_expansion @ self.small_model.patch_embed.proj.weight.reshape(
            self.D1, -1
        )
        self.full_dict["patch_embed.proj.weight"] = self.full_dict[
            "patch_embed.proj.weight"
        ].reshape(self.D2, 3, 16, 16)
        self.large_model.patch_embed.proj.weight.data.copy_(
            self.full_dict["patch_embed.proj.weight"]
        )
        self.full_dict["patch_embed.proj.bias"] = (
            self.embedding_expansion @ self.small_model.patch_embed.proj.bias
        )
        self.large_model.patch_embed.proj.bias.data.copy_(
            self.full_dict["patch_embed.proj.bias"]
        )

        self.full_dict["head.weight"] = (
            self.small_model.head.weight @ self.embedding_expansion.T
        )
        self.large_model.head.weight.data.copy_(self.full_dict["head.weight"])
        self.full_dict["head.bias"] = self.small_model.head.bias
        self.large_model.head.bias.data.copy_(self.small_model.head.bias)

        # Class token
        self.full_dict["cls_token"] = (
            self.small_model.cls_token @ self.embedding_expansion.T
        )
        self.large_model.cls_token.data.copy_(self.full_dict["cls_token"])
        # Outer encoder Fn
        self.full_dict["norm.weight"] = (
            self.embedding_expansion @ self.small_model.norm.weight
        )
        self.large_model.norm.weight.data.copy_(self.full_dict["norm.weight"])
        self.full_dict["norm.bias"] = (
            self.embedding_expansion @ self.small_model.norm.bias
        )
        self.large_model.norm.bias.data.copy_(self.full_dict["norm.bias"])

    def forward(self, X: Tensor) -> Tensor:
        if self.enable_ligo:
            self.expand_params()
            out = self.large_model.forward(X)
        else:
            out = self.small_model.forward(X)
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
                if name != "pos_embed":
                    if "blocks" in name:
                        out_layer_key = ".".join(name.split(".")[:2])
                        inner_layer_key = ".".join(name.split(".")[2:])
                        self.full_dict[out_layer_key][inner_layer_key].backward(
                            param.grad, retain_graph=True
                        )
                    else:
                        if name != "head.bias":
                            self.full_dict[name].backward(param.grad, retain_graph=True)
        else:
            pass

    def save_large_model(self, model_path: str):
        self.large_model._save_to_state_dict(model_path)

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.embedding_expansion)

        for key in self.width_expansion_operator.keys():
            nn.init.xavier_normal_(self.width_expansion_operator[key])

        nn.init.xavier_normal_(self.depth_expansion_operator)

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


# For test
if __name__ == "__main__":
    params = dict(
        patch_size=16,
        n_hiddens=192,
        n_layers=12,
        num_heads=3,
        target_hiddens=384,
        target_layers=12,
        target_heads=4,
        small_model=None,
        num_classes=100,
    )
    ligo = LiGOViT(**params)
    for name, param in ligo.named_parameters():
        print(name, param.size())

    params = dict(
        patch_size=16,
        n_hiddens=192,
        n_layers=12,
        num_heads=3,
        small_model_path=None,
        num_classes=100,
    )
    ligo = LiGOViT(**params)
    for name, param in ligo.named_parameters():
        print(name, param.size())
