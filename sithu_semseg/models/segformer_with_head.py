import torch
from torch import Tensor
from torch.nn import functional as F
from sithu_semseg.models.base import BaseModel
from sithu_semseg.models.heads import SegFormerHead

import torch.nn as nn

def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = nn.Sigmoid()
        super().__init__(conv2d, upsampling, activation)

class SegFormerWithHead(BaseModel):
    def __init__(self, backbone: str = 'MiT-B0', num_classes: int = 19) -> None:
        super().__init__(backbone, num_classes)
        self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 768, 128)
        self.apply(self._init_weights)
        self.segmentation_head = SegmentationHead(128, num_classes, 1, upsampling=4)
        initialize_head(self.segmentation_head)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = self.segmentation_head(y)
        # y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape

        # y = torch.relu(y)
        return y


if __name__ == '__main__':
    model = SegFormer('MiT-B0', num_classes=1)
    # model.load_state_dict(torch.load('checkpoints/pretrained/segformer/segformer.b0.ade.pth', map_location='cpu'))
    x = torch.rand(1, 3, 512, 512)
    y = model(x)
    print(y)
    print(y.shape)
