import torch
import torch.nn as nn


from functools import partial
from typing import List

import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers.squeeze_excite import EffectiveSEModule
from timm.models import named_apply
from timm.layers import DropPath
from timm.layers import LayerNorm2d

class RDNetClassifierHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        drop_rate: float = 0.,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_features = in_features

        self.norm = nn.LayerNorm(in_features)
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def reset(self, num_classes):
        self.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, pre_logits: bool = False):
        x = x.mean([-2, -1])
        x = self.norm(x)
        x = self.drop(x)
        if pre_logits:
            return x
        x = self.fc(x)
        return x


class PatchifyStem(nn.Module):
    def __init__(self, num_input_channels, num_init_features, patch_size=4):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(num_input_channels, num_init_features, kernel_size=patch_size, stride=patch_size),
            LayerNorm2d(num_init_features),
        )

    def forward(self, x):
        return self.stem(x)


class Block(nn.Module): #this is Feature MIXER
    """D == Dw conv, N == Norm, F == Feed Forward, A == Activation"""
    def __init__(self, in_chs, inter_chs, out_chs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chs, in_chs, groups=in_chs, kernel_size=7, stride=1, padding=3),
            LayerNorm2d(in_chs, eps=1e-6),
            nn.Conv2d(in_chs, inter_chs, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(inter_chs, out_chs, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.layers(x)


class BlockESE(nn.Module): # this is Feature MIXER
    """D == Dw conv, N == Norm, F == Feed Forward, A == Activation"""
    def __init__(self, in_chs, inter_chs, out_chs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chs, in_chs, groups=in_chs, kernel_size=7, stride=1, padding=3),
            LayerNorm2d(in_chs, eps=1e-6),
            nn.Conv2d(in_chs, inter_chs, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(inter_chs, out_chs, kernel_size=1, stride=1, padding=0),
            EffectiveSEModule(out_chs),
        )

    def forward(self, x):
        return self.layers(x)


class DenseBlock(nn.Module):
    def __init__(
        self,
        num_input_features,
        growth_rate,
        bottleneck_width_ratio, # I think this is wide part of bottleneck
        drop_path_rate,
        drop_rate=0.0,
        rand_gather_step_prob=0.0,
        block_idx=0,
        block_type="Block",
        ls_init_value=1e-6,
        **kwargs,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.rand_gather_step_prob = rand_gather_step_prob
        self.block_idx = block_idx
        self.growth_rate = growth_rate

        self.gamma = nn.Parameter(ls_init_value * torch.ones(growth_rate)) if ls_init_value > 0 else None
        growth_rate = int(growth_rate)
        inter_chs = int(num_input_features * bottleneck_width_ratio / 8) * 8

        if self.drop_path_rate > 0:
            self.drop_path = DropPath(drop_path_rate)

        self.layers = eval(block_type)(
            in_chs=num_input_features,
            inter_chs=inter_chs,
            out_chs=growth_rate, # the concatentation of features
        )

    def forward(self, x):
        if isinstance(x, List):
            x = torch.cat(x, 1)
        x = self.layers(x)

        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))

        if self.drop_path_rate > 0 and self.training:
            x = self.drop_path(x)
        return x


class DenseStage(nn.Sequential):
    def __init__(self, num_block, num_input_features, drop_path_rates, growth_rate, **kwargs):
        super().__init__()
        for i in range(num_block):
            layer = DenseBlock(
                num_input_features=num_input_features,
                growth_rate=growth_rate,
                drop_path_rate=drop_path_rates[i],
                block_idx=i,
                **kwargs,
            )
            num_input_features += growth_rate
            self.add_module(f"dense_block{i}", layer)
        self.num_out_features = num_input_features

    def forward(self, init_feature):
        features = [init_feature]
        for module in self:
            new_feature = module(features)
            features.append(new_feature)
        return torch.cat(features, 1)


class RDNet(nn.Module):
    def __init__(
        self,
        num_init_features=64,
        growth_rates=(64, 104, 128, 128, 128, 128, 224),
        num_blocks_list=(3, 3, 3, 3, 3, 3, 3),
        bottleneck_width_ratio=4,
        zero_head=False,
        in_chans=3,  # timm option [--in-chans]
        num_classes=1000,  # timm option [--num-classes]
        drop_rate=0.0,  # timm option [--drop: dropout ratio]
        drop_path_rate=0.0,  # timm option [--drop-path: drop-path ratio]
        checkpoint_path=None,  # timm option [--initial-checkpoint]
        transition_compression_ratio=0.5,
        ls_init_value=1e-6,
        is_downsample_block=(None, True, True, False, False, False, True),
        block_type="Block",
        head_init_scale: float = 1.,
        **kwargs,
    ):
        super().__init__()
        assert len(growth_rates) == len(num_blocks_list) == len(is_downsample_block)

        self.num_classes = num_classes
        if isinstance(block_type, str):
            block_type = [block_type] * len(growth_rates)

        # stem
        self.stem = PatchifyStem(in_chans, num_init_features, patch_size=4)

        # features
        self.feature_info = []
        self.num_stages = len(growth_rates)
        curr_stride = 4  # stem_stride
        num_features = num_init_features
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(num_blocks_list)).split(num_blocks_list)]

        dense_stages = []
        for i in range(self.num_stages):
            dense_stage_layers = []
            if i != 0:
                compressed_num_features = int(num_features * transition_compression_ratio / 8) * 8
                k_size = stride = 1
                if is_downsample_block[i]:
                    curr_stride *= 2
                    k_size = stride = 2
                dense_stage_layers.append(LayerNorm2d(num_features))
                dense_stage_layers.append(
                    nn.Conv2d(num_features, compressed_num_features, kernel_size=k_size, stride=stride, padding=0)
                )
                num_features = compressed_num_features

            stage = DenseStage(
                num_block=num_blocks_list[i],
                num_input_features=num_features,
                growth_rate=growth_rates[i],
                bottleneck_width_ratio=bottleneck_width_ratio,
                drop_rate=drop_rate,
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                block_type=block_type[i],
            )
            dense_stage_layers.append(stage)
            num_features += num_blocks_list[i] * growth_rates[i]

            if i + 1 == self.num_stages or (i + 1 != self.num_stages and is_downsample_block[i + 1]):
                self.feature_info += [
                    dict(
                        num_chs=num_features,
                        reduction=curr_stride,
                        module=f'dense_stages.{i}',
                        growth_rate=growth_rates[i],
                    )
                ]
            dense_stages.append(nn.Sequential(*dense_stage_layers))
        self.dense_stages = nn.Sequential(*dense_stages)

        # classifier
        self.head = RDNetClassifierHead(num_features, num_classes, drop_rate=drop_rate)

        # initialize weights
        named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

        if zero_head:
            nn.init.zeros_(self.head[-1].weight.data)
            if self.head[-1].bias is not None:
                nn.init.zeros_(self.head[-1].bias.data)

        if checkpoint_path is not None:
            self.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes=0, global_pool=None):
        assert global_pool is None
        self.head.reset(num_classes)

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.dense_stages(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def group_matcher(self, coarse=False):
        assert not coarse
        return dict(
            stem=r'^stem',
            blocks=r'^dense_stages\.(\d+)',
        )


def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.constant_(module.bias, 0)
        if name and 'head.' in name:
            module.weight.data.mul_(head_init_scale)
            module.bias.data.mul_(head_init_scale)


def rdnet_tiny(pretrained=False, num_classes=1000, checkpoint_path=None, device="cpu", **kwargs):
    n_layer = 7
    model_args = {
        "num_init_features": 64,
        "growth_rates": [64, 104, 128, 128, 128, 128, 224],
        "num_blocks_list": [3] * n_layer,
        "is_downsample_block": (None, True, True, False, False, False, True),
        "transition_compression_ratio": 0.5,
        "block_type": ["Block", "Block", "BlockESE", "BlockESE", "BlockESE", "BlockESE", "BlockESE"],
        "num_classes": num_classes,
    }

    model = RDNet(**{**model_args, **kwargs})

    if pretrained:
        assert checkpoint_path is not None, "Please provide checkpoint_path for pretrained weights"
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)

    return model