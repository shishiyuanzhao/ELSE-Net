from tqdm import tqdm
from itertools import repeat
import collections.abc
from typing import Tuple, Union, Callable, Optional
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict

from torch import nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import os
import clip
import numpy as np
import open_clipn
from open_clip.transform import image_transform
import json
import random
from collections import defaultdict
import shutil
from typing import List
from detectron2.data import MetadataCatalog
from ..model.clip_utils.utils import get_labelset_from_dataset
from open_clip.transformer import VisionTransformer
from ..model.attn_helper import cross_attn_layer, downsample2d, resize_pos_embed2d
from detectron2.layers import ShapeSpec


def torch_save(classifer, save_path="./"):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(classifer.cpu(), f)
        
def torch_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier
    
def get_clipn_classifier(dataset_name: str, model,  device):
        txt = []
        category_names = get_labelset_from_dataset(dataset_name)
        N=len(category_names)
        model.eval()

        if N:
            with open("/root/autodl-tmp/SAN-main/prompt_no.txt") as f:
                prompt_lis = f.readlines()
            num_prom = len(prompt_lis)            
        for idx in range(num_prom):
            for name in category_names:
                txt.append(open_clipn.tokenize(prompt_lis[idx].replace("\n", "").format(name), 77).unsqueeze(0))
        txt = torch.cat(txt, dim=0)
        txt = txt.reshape(num_prom, len(category_names), -1)
        text_inputs = txt.cuda()
        
        text_yes_ttl = torch.zeros(len(category_names), 512).to(device)
        text_no_ttl = torch.zeros(len(category_names), 512).to(device)
        
        with torch.no_grad():
            for i in range(num_prom):
                text_yes_i = model.encode_text(text_inputs[i])
                text_yes_i = F.normalize(text_yes_i, dim=-1)
                text_no_i = model.encode_text(text_inputs[i], "no")
                text_no_i = F.normalize(text_no_i, dim=-1)
                
                text_yes_ttl += text_yes_i
                text_no_ttl += text_no_i
            
        weight_no = F.normalize(text_no_ttl, dim=-1)
        weight_yes = F.normalize(text_yes_ttl, dim=-1)
        bg_embed = nn.Parameter(torch.randn(1, weight_no.shape[1])).to(device)
        weight_no = torch.cat([weight_no,bg_embed],dim=0)
        weight_yes = torch.cat([weight_yes,bg_embed],dim=0)
        weight_no = 100 * weight_no
        weight_yes = 100 *  weight_yes
        yesno = torch.cat([weight_no.unsqueeze(-1), weight_yes.unsqueeze(-1) ], -1)
        probs_no = torch.softmax(yesno, dim=-1)[:,:,1]
        # clip_classifier = ViT_Classifier(VisualTransformer, weight_yes, weight_no)
        return weight_no

    
class ViT_Classifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head_yes, classification_head_no):
        super().__init__()
        self.image_encoder = image_encoder
        flag = True
        self.fc_yes = nn.Parameter(classification_head_yes, requires_grad=flag)    # num_classes  num_feat_dimension
        self.fc_no = nn.Parameter(classification_head_no, requires_grad=flag)      # num_classes  num_feat_dimension
        self.scale = 100. # this is from the parameter of logit scale in CLIPN
        
    def set_frozen(self, module):
        for module_name in module.named_parameters():
            module_name[1].requires_grad = False
    def set_learnable(self, module):
        for module_name in module.named_parameters():
            module_name[1].requires_grad = True
            
    def forward(self, x):
        inputs = self.image_encoder(x)
        inputs_norm = F.normalize(inputs, dim=-1)
        fc_yes = F.normalize(self.fc_yes, dim=-1)
        fc_no = F.normalize(self.fc_no, dim=-1)
        
        logits_yes = self.scale  * fc_yes.cuda()
        logits_no = self.scale * fc_no.cuda()

        yesno = torch.cat([logits_yes.unsqueeze(-1), logits_no.unsqueeze(-1) ], -1)
        probs_no = torch.softmax(yesno, dim=-1)[:,:,1]
        
        return probs_no
    
    def save(self, path = "./"):
        torch_save(self, path)
        
    @classmethod
    def load(cls, filename = "./", device=None):
        return torch_load(filename, device)
    

class VisualTransformer(nn.Module):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            output_dim: int,
            act_layer: Callable = nn.GELU
    ):
        super().__init__()
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, mlp_ratio, act_layer=act_layer)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False


    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        num_imgtoken = x.size(1)
        num_postoken = self.positional_embedding.size(0)
        if num_imgtoken == num_postoken:
            x = x + self.positional_embedding.to(x.dtype)
        else:
            ori_hw, new_hw = int((num_postoken - 1)**0.5), int((num_imgtoken - 1)**0.5)
            img_pos_embedding = self.positional_embedding[1:].reshape(ori_hw, ori_hw, -1).permute(2,0,1).unsqueeze(0)
            img_pos_embedding = F.interpolate(img_pos_embedding, size=(new_hw, new_hw), mode='bicubic')
            img_pos_embedding = img_pos_embedding[0].permute(1,2,0).reshape(new_hw*new_hw, -1)
            new_pos_embedding = torch.cat([self.positional_embedding[0].unsqueeze(0), img_pos_embedding],dim=0)
            x = x + new_pos_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x0 = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x_yes = x0 @ self.proj

        return x_yes#, self.ln_post(x[:, 1:, :]) @ self.proj, self.proj
    
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)
    
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int,  mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x
    
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            act_layer: Callable = nn.GELU,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()

        self.ln_1 = LayerNorm(d_model)
        # FIXME torchscript issues need to be resolved for custom attention
        # if scale_cosine_attn or scale_heads:
        #     self.attn = Attention(
        #        d_model, n_head,
        #        scaled_cosine=scale_cosine_attn,
        #        scale_heads=scale_heads,
        #     )
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_attn = LayerNorm(d_model) if scale_attn else nn.Identity()

        self.ln_2 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ('ln', LayerNorm(mlp_width) if scale_fc else nn.Identity()),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        # FIXME torchscript issues need resolving for custom attention option to work
        # if self.use_torch_attn:
        #     return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        # else:
        #     return self.attn(x, attn_mask=attn_mask)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ln_attn(self.attention(self.ln_1(x), attn_mask=attn_mask))
        x = x + self.mlp(self.ln_2(x))
        return x
