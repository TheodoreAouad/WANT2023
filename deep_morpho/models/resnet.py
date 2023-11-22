"""Some code taken from torch source code <https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py>"""

import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
try:
    from torchvision.models.utils import load_state_dict_from_url
except ModuleNotFoundError:
    from torch.hub import load_state_dict_from_url

from .binary_nn import BinaryNN


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

model_layers = {
    'resnet50': [3, 4, 6, 3],  # difference between 50 and 34 is the block
    'resnet34': [3, 4, 6, 3],
    'resnet18': [2, 2, 2, 2],
    'resnet10': [1, 1, 1, 1],
}

model_blocks = {
    'resnet18': resnet.BasicBlock,
    'resnet34': resnet.BasicBlock,
    'resnet50': resnet.Bottleneck,
}


class PreResNet(resnet.ResNet, BinaryNN):
    r""" ResNet model from torch, modified to have more flexibility with layers. From
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Use ResNet class defined below..
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    def __init__(
        self,
        n_classes=1000,
        layers=[2, 2, 2, 2],
        planes=[64, 128, 256, 512],
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        block=resnet.BasicBlock,
        **kwargs
    ):
        BinaryNN.__init__(self)
        self.layers = layers
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead

            replace_stride_with_dilation = [False] * (len(layers) - 1)

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, planes[0], layers[0])
        for layer_idx in range(1, len(layers)):
            setattr(
                self,
                'layer{}'.format(layer_idx+1),
                self._make_layer(block, planes[layer_idx], layers[layer_idx], stride=2,
                                dilate=replace_stride_with_dilation[layer_idx-1])
            )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.inplanes * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(len(self.layers)):
            x = getattr(self, 'layer{}'.format(i+1))(x)


        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    @property
    def device(self):
        for p in self.parameters():
            return p.device

    def _get_resolution(self, input_res, filter_size, stride, padding):
        return int((input_res - filter_size + 2*padding) / stride) + 1


class ResNet(PreResNet):
    """Resnet class to use. Allows to:
        - say which type of resnet we want (18, 34 or 50)
        - have different in channels
        - pretraining
        Replaces the function "torchvision.models.resnetX"
    """

    def __init__(
        self,
        in_channels=1,
        n_classes=1,
        pretrained=False,
        progress=True,
        activation_fn=lambda x: torch.softmax(x, dim=-1),
        do_activation=False,
        layers='resnet18',
        planes=[64, 128, 256, 512],
        block=resnet.BasicBlock,
        **kwargs
    ):
        if type(layers) == str:
            model_name = layers
            block = model_blocks[layers]
            layers = model_layers[layers]
        else:
            pretrained = False
        super().__init__(
            layers=layers,
            planes=planes,
            block=block,
            **kwargs
        )

        self.n_classes = n_classes
        self.in_channels = in_channels

        if type(pretrained) == bool and pretrained:
            state_dict = load_state_dict_from_url(model_urls[model_name],
                                                progress=progress)
            self.load_state_dict(state_dict)

        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3,
                            bias=False)

        self.fc = nn.Linear(self.inplanes * resnet.BasicBlock.expansion, n_classes)

        self.do_activation = do_activation
        self.final_activation = activation_fn

        if type(pretrained) == str and pretrained:
            self.load_state_dict(torch.load(pretrained))

    def forward(self, x, do_activation=None):
        if do_activation is None:
            act = self.do_activation
        else:
            act = do_activation
        x = super().forward(x)
        if act:
            return self.final_activation(x)
        return x


class ResNet18(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(layers='resnet18', *args, **kwargs)


class ResNet34(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(layers='resnet34', *args, **kwargs)


class ResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(layers='resnet50', *args, **kwargs)


# class ResNet101(ResNet):
#     def __init__(self, *args, **kwargs):
#         super().__init__(layers='resnet101', *args, **kwargs)


# class ResNet152(ResNet):
#     def __init__(self, *args, **kwargs):
#         super().__init__(layers='resnet152', *args, **kwargs)
