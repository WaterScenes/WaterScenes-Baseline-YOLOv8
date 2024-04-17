import torch
import torch.nn as nn
import math
import torchvision.ops


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))

        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x


class RadarConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, first_calculator='pool'):
        super(RadarConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if first_calculator == 'conv':
            self.initial_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                          stride=stride, padding=kernel_size // 2)
        elif first_calculator == 'pool':
            self.initial_conv = nn.AvgPool2d(3, stride=1, padding=1)

        self.deformable_conv = DeformableConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                                stride=stride, padding=3 // 2)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.norm(self.deformable_conv(x))
        return x


class RCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super(RCBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.radar_conv = RadarConv(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1)

        if down is False:
            self.weight_conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                          padding=0)
        else:
            self.weight_conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2,
                                          padding=1)

        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x):
        x_res = x
        x = self.radar_conv(x)
        x = self.weight_conv2(x)
        x = self.norm(x)
        x = self.activation(x)

        return x

class RCNet(nn.Module):
    def __init__(self, in_channels, width=0.75):
        super(RCNet, self).__init__()
        self.in_channels = in_channels

        base_channels = [int(item*width*0.25) for item in [64, 128, 256, 512, 1024]]

        stage_blocks = []
        for i in range(len(base_channels)):
            if i == 0:
                stage_blocks.append(RCBlock(in_channels=in_channels,
                                            out_channels=base_channels[i], down=True))
            else:
                stage_blocks.append(RCBlock(in_channels=base_channels[i-1],
                                            out_channels=base_channels[i], down=True))

        self.rc_blocks = nn.ModuleList(stage_blocks)
        # print(len(self.rc_blocks))

    def forward_blocks(self, x):
        output_features = []
        for i, block in enumerate(self.rc_blocks):
            x = block(x)
            if i > 1:
                output_features.append(x)
        return output_features

    def forward(self, x):
        x = self.forward_blocks(x)
        return x

if __name__ == '__main__':
    input_map = torch.randn(1, 4, 640, 640)
    model = RCNet(in_channels=4)
    output_map = model(input_map)
    print(output_map[0].shape)
    print(output_map[1].shape)
    print(output_map[2].shape)