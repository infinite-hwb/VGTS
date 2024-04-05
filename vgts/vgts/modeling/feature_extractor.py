from itertools import chain

import torch

from torchvision.models.resnet import ResNet, resnet50, resnet101

from vgts.structures.feature_map import FeatureMapSize


GROUPNORM_NUMGROUPS = 32


def build_feature_extractor(backbone_arch, use_group_norm=False):
    if backbone_arch.lower() == "resnet50":
        net = resnet50_c4(use_group_norm=use_group_norm)
    elif backbone_arch.lower() == "resnet101":
        net = resnet101_c4(use_group_norm=use_group_norm)
    else:
        raise(RuntimeError("Unknown backbone arch: {0}".format(backbone_arch)))
    return net


class ResNetFeatureExtractor(ResNet):
    def __init__(self, resnet_full, level,
                       feature_map_stride, feature_map_receptive_field):
        self.__dict__ = resnet_full.__dict__.copy()
        self.feature_map_receptive_field = feature_map_receptive_field
        self.feature_map_stride = feature_map_stride
        self._feature_level = level
        delattr(self, "fc")
        delattr(self, "avgpool")

        assert level in [1, 2, 3, 4, 5], "Feature level should be one of 1, 2, 3, 4, 5"
        self.resnet_blocks = [self.layer1, self.layer2, self.layer3, self.layer4]
        layer_names = ["layer1", "layer2", "layer3", "layer4"]

        self.resnet_blocks = self.resnet_blocks[:level-1]
        for name in layer_names[level-1:]:
            delattr(self, name)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.resnet_blocks:
            x = layer(x)
        return x

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()

    def freeze_blocks(self, num_blocks=0):
        layer0 = [torch.nn.ModuleList([self.conv1, self.bn1])]

        num_remaining_blocks = num_blocks
        blocks = chain(layer0, chain.from_iterable(self.resnet_blocks))
        for b in blocks:
            if num_remaining_blocks > 0:
                self.freeze_layer_parameters(b)
                num_remaining_blocks -= 1

    @staticmethod
    def freeze_layer_parameters(layer):
        for p in layer.parameters():
            p.requires_grad = False

    def get_num_blocks_in_feature_extractor(self):
        num_blocks = 1 + sum(len(b) for b in self.resnet_blocks)
        return num_blocks


def get_norm_layer(use_group_norm):
    if use_group_norm:
        return lambda width: torch.nn.GroupNorm(GROUPNORM_NUMGROUPS, width)
    else:
        return torch.nn.BatchNorm2d


def _resnet_fe(resnet, level, use_group_norm, feature_map_stride, feature_map_receptive_field):
    return ResNetFeatureExtractor(resnet(norm_layer=get_norm_layer(use_group_norm)), level,
                                  feature_map_stride, feature_map_receptive_field)


def resnet50_c4(use_group_norm=False):
    return _resnet_fe(resnet50, 4, use_group_norm,
                      feature_map_stride=FeatureMapSize(h=16, w=16),
                      feature_map_receptive_field=FeatureMapSize(h=16, w=16))


def resnet101_c4(use_group_norm=False):
    return _resnet_fe(resnet101, 4, use_group_norm,
                      feature_map_stride=FeatureMapSize(h=16, w=16),
                      feature_map_receptive_field=FeatureMapSize(h=16, w=16))

