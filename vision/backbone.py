"""ResNet18 frame features shared by the independent RGB and depth experiments.

This is a dependency-light adaptation of the supplied MVTN ResNet18 backbone
(`models/mvtn/src_mvtn/models/backbones/resnet.py`). It retains the eight basic
residual blocks and MVTN's Dropout2d placement. Setting dropout to zero gives
the standard ResNet18 feature extractor used by the TSM adapter.

TSM shifts channels immediately before each residual branch's first 3x3
convolution, following Ji Lin, Chuang Gan and Song Han's `blockres` version in
`models/tsm/ops/temporal_shift.py` (upstream MIT license in models/tsm/LICENSE).
The identity branch is not shifted, and shift boundaries never cross clips.
There are no torchvision imports, automatic downloads, or dataset writes.
"""
import torch
from torch import nn


def temporal_shift(x, num_frames, fold_div=8):
    """Shift one channel group forward, one backward; zero clip boundaries."""
    if x.ndim != 4 or type(num_frames) is not int or num_frames < 1:
        raise ValueError("Temporal shift needs [clips*frames,C,H,W] and positive frame count")
    if x.shape[0] % num_frames or type(fold_div) is not int or fold_div < 2:
        raise ValueError("Invalid temporal shift clip length or channel division")
    nt, channels, height, width = x.shape
    fold = channels // fold_div
    if fold == 0:
        raise ValueError("Temporal shift division exceeds the available channels")
    clips = x.reshape(nt // num_frames, num_frames, channels, height, width)
    shifted = torch.zeros_like(clips)
    shifted[:, :-1, :fold] = clips[:, 1:, :fold]
    shifted[:, 1:, fold:2 * fold] = clips[:, :-1, fold:2 * fold]
    shifted[:, :, 2 * fold:] = clips[:, :, 2 * fold:]
    return shifted.reshape_as(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dropout=0., num_frames=None, shift_div=8):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout2d(dropout)
        self.num_frames = num_frames
        self.shift_div = shift_div
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))

    def forward(self, x):
        identity = x
        branch = temporal_shift(x, self.num_frames, self.shift_div) if self.num_frames else x
        out = self.dropout(self.relu(self.bn1(self.conv1(branch))))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
            # This extra dropout is present in the supplied MVTN basic block.
            out = self.dropout(out)
        return self.dropout(self.relu(out + identity))


class ResNet18Features(nn.Module):
    """Return 512 features per frame; state keys match a standard ResNet18."""

    def __init__(self, in_channels=3, dropout=0., num_frames=None, shift_div=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        widths = [64, 128, 256, 512]
        current = 64
        for index, width in enumerate(widths, 1):
            stride = 1 if index == 1 else 2
            layer = nn.Sequential(BasicBlock(current, width, stride, dropout, num_frames, shift_div),
                                  BasicBlock(width, width, 1, dropout, num_frames, shift_div))
            setattr(self, 'layer' + str(index), layer)
            current = width
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
        return self.avgpool(x).flatten(1)
