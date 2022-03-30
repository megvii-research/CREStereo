"""
Feature extractor based on "RAFT: Recurrent All Pairs Field Transforms for Optical Flow".
Modified from: https://github.com/princeton-vl/RAFT/blob/master/core/extractor.py
"""


import megengine.module as M
import megengine.functional as F


class ResidualBlock(M.Module):
    def __init__(self, in_planes, planes, norm_fn="group", stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = M.Conv2d(
            in_planes, planes, kernel_size=3, padding=1, stride=stride
        )
        self.conv2 = M.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = M.ReLU()

        num_groups = planes // 8

        if norm_fn == "group":
            self.norm1 = M.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = M.GroupNorm(num_groups=num_groups, num_channels=planes)
            norm3 = M.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == "batch":
            self.norm1 = M.BatchNorm2d(planes)
            self.norm2 = M.BatchNorm2d(planes)
            norm3 = M.BatchNorm2d(planes)

        elif norm_fn == "instance":
            self.norm1 = M.InstanceNorm(planes, affine=False)
            self.norm2 = M.InstanceNorm(planes, affine=False)
            norm3 = M.InstanceNorm(planes, affine=False)

        elif norm_fn == "none":
            self.norm1 = M.Sequential()
            self.norm2 = M.Sequential()
            norm3 = M.Sequential()

        self.downsample = M.Sequential(
            M.Conv2d(in_planes, planes, kernel_size=1, stride=stride), norm3
        )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        x = self.downsample(x)

        return self.relu(x + y)


class BasicEncoder(M.Module):
    def __init__(self, output_dim=128, norm_fn="batch", dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == "group":
            self.norm1 = M.GroupNorm(num_groups=8, num_channels=64)

        elif self.norm_fn == "batch":
            self.norm1 = M.BatchNorm2d(64)

        elif self.norm_fn == "instance":
            self.norm1 = M.InstanceNorm(64, affine=False)

        elif self.norm_fn == "none":
            self.norm1 = M.Sequential()

        self.conv1 = M.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = M.ReLU()

        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=1)

        self.conv2 = M.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = M.Dropout(drop_prob=dropout)

        for m in self.modules():
            if isinstance(m, M.Conv2d):
                M.init.msra_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (M.BatchNorm2d, M.InstanceNorm, M.GroupNorm)):
                if m.weight is not None:
                    M.init.fill_(m.weight, 1)
                if m.bias is not None:
                    M.init.fill_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return M.Sequential(*layers)

    def forward(self, x):

        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = F.concat(x, axis=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = F.split(x, 2, axis=0)

        return x
