import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
import numpy as np
from beit3_tools import utils
from CL_framework.config import  _get_base_config, _get_large_config
from beit3_tools.modeling_finetune import TwoLayerMLP, Pooler
from CL_tools.losses import SupConLoss
from beit3_tools.modeling_utils import BEiT3Wrapper

class BEiT3ForImageClassification(BEiT3Wrapper):
    def __init__(
            self,
            args,
            num_classes,
            norm_layer=nn.LayerNorm,
            **kwargs
    ):
        super(BEiT3ForImageClassification, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.fc_norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.fc_norm.apply(self._init_weights)
        self.head.apply(self._init_weights)
        init_scale = 0.001
        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)
        dim_in = 768
        feat_dim = 128
        self.cl_head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        self.cl_head.apply(self._init_weights)

    def forward(self, image, **kwargs):
        x = self.beit3(textual_tokens=None, visual_tokens=image)["encoder_out"]
        t = x[:, 1:, :]
        cls_x = self.fc_norm(t.mean(1))
        return self.head(cls_x), self.cl_head(cls_x), cls_x




@register_model
def beit3_base_patch16_224_imageclassification(pretrained=False, **kwargs):
    args = _get_base_config(**kwargs)
    args.normalize_output = False
    model = BEiT3ForImageClassification(args, num_classes=args.n_cls, **kwargs)
    return model


@register_model
def beit3_large_patch16_224_imageclassification(pretrained=False, **kwargs):
    args = _get_large_config(**kwargs)
    args.normalize_output = False
    model = BEiT3ForImageClassification(args, num_classes=args.n_cls, **kwargs)
    return model

