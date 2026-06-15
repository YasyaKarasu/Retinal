
from functools import partial
from types import MethodType

import timm.models.vision_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.models.layers import trunc_normal_

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1,keepdim=True)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def RETFound_mae(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



def Dinov2(args, **kwargs):
    
    if args.model_arch == 'dinov2_vits14':
        arch = 'vit_small_patch14_dinov2.lvd142m'
    elif args.model_arch == 'dinov2_vitb14':
        arch = 'vit_base_patch14_dinov2.lvd142m'
    elif args.model_arch == 'dinov2_vitl14':
        arch = 'vit_large_patch14_dinov2.lvd142m'
    elif args.model_arch == 'dinov2_vitg14':
        arch = 'vit_giant_patch14_dinov2.lvd142m'
    else:
        raise ValueError(f"Unknown model_arch '{args.model_arch}'. "
                         f"Expected one of: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14")
        
    model = timm.create_model(
        arch,
        pretrained=True,
        img_size=args.input_size,
        **kwargs
    )
    return model



def RETFound_dinov2(args, **kwargs):
    if args.input_size % 14 != 0:
        raise ValueError(
            "RETFound-DINOv2 input size must be divisible by patch size 14"
        )
    model = timm.create_model(
        'vit_large_patch14_dinov2.lvd142m',
        pretrained=False,
        img_size=args.input_size,
        **kwargs
    )
    return model


def add_challenge_head(model, num_classes=28):
    """Attach a second classifier while preserving the existing head keys."""
    if not hasattr(model, "head") or not isinstance(model.head, nn.Linear):
        raise TypeError(
            "Dual-head training requires a model with a linear `head`"
        )
    if not hasattr(model, "forward_features") or not hasattr(
        model, "forward_head"
    ):
        raise TypeError(
            "Dual-head training requires `forward_features` and `forward_head`"
        )

    model.challenge_head = nn.Linear(model.head.in_features, num_classes)
    trunc_normal_(model.challenge_head.weight, std=2e-5)
    if model.challenge_head.bias is not None:
        nn.init.zeros_(model.challenge_head.bias)

    def forward_with_challenge_head(self, x):
        features = self.forward_features(x)
        pre_logits = self.forward_head(features, pre_logits=True)
        return {
            "all_classes": self.head(pre_logits),
            "challenge28": self.challenge_head(pre_logits),
        }

    model.forward = MethodType(forward_with_challenge_head, model)
    return model


def Dinov3(args, **kwargs):
    # Load ViT-L/16 backbone (hub model has `head = Identity` by default)
    model = torch.hub.load(
        repo_or_dir="facebookresearch/dinov3",
        model=args.model_arch,
        pretrained=False,   # main() will load your checkpoint
        trust_repo=True,
    )

    # Figure out feature dimension for the probe
    feat_dim = getattr(model, "embed_dim", None) or getattr(model, "num_features", None)
    model.head = nn.Linear(feat_dim, args.nb_classes)
    trunc_normal_(model.head.weight, std=2e-5)
    if model.head.bias is not None:
        nn.init.zeros_(model.head.bias)

    return model
