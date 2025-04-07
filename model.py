# Fixed model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
FEATURE_CHANNELS = 64

class MobileNetV3FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV3FeatureExtractor, self).__init__()

        if pretrained:
            weights = MobileNet_V3_Small_Weights.DEFAULT
        else:
            weights = None

        mobilenet = mobilenet_v3_small(weights=weights)
        self.features = nn.Sequential(*list(mobilenet.features.children())[:6])

        if pretrained:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x, scale=4):
        x = self.features(x)
        B, C, H, W = x.shape
        x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class TemporalConsistencyModule(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super(TemporalConsistencyModule, self).__init__()
        self.conv_h = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv_x = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv_o = nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)

        self.flow_conv1 = nn.Conv2d(in_channels * 2, 24, kernel_size=3, padding=1)
        self.flow_conv2 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.flow_conv3 = nn.Conv2d(24, 2, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, current, previous=None):
        if previous is None:
            return current

        flow_input = torch.cat([current, previous], dim=1)
        flow = self.relu(self.flow_conv1(flow_input))
        flow = self.relu(self.flow_conv2(flow))
        flow = self.flow_conv3(flow)

        b, c, h, w = previous.size()
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=2).float().to(current.device)
        grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)
        flow_grid = grid + flow.permute(0, 2, 3, 1)

        flow_grid[:, :, :, 0] = 2.0 * flow_grid[:, :, :, 0] / (w - 1) - 1.0
        flow_grid[:, :, :, 1] = 2.0 * flow_grid[:, :, :, 1] / (h - 1) - 1.0

        warped_previous = F.grid_sample(previous, flow_grid, mode='bilinear', padding_mode='border', align_corners=True)

        h_state = self.conv_h(warped_previous)
        x_state = self.conv_x(current)
        combined = self.relu(h_state + x_state)
        out = self.conv_o(combined)

        return out

class ReconstructionModule(nn.Module):
    def __init__(self, in_channels, scale_factor=4):
        super(ReconstructionModule, self).__init__()
        self.scale_factor = scale_factor
        num_upscales = int(math.log2(scale_factor))
        assert 2 ** num_upscales == scale_factor, "scale_factor must be a power of 2"

        self.conv1 = nn.Conv2d(in_channels, FEATURE_CHANNELS, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(FEATURE_CHANNELS) for _ in range(5)])
        print("num_upscales=", num_upscales)
        up_layers = []
        for _ in range(num_upscales):
            up_layers += [
                nn.Conv2d(FEATURE_CHANNELS, FEATURE_CHANNELS * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            ]
        self.upscale = nn.Sequential(*up_layers)
        self.final_conv = nn.Conv2d(FEATURE_CHANNELS, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_blocks(x)
        x = self.upscale(x)
        x = self.final_conv(x)
        return x

class VideoSuperResolution(nn.Module):
    def __init__(self, scale_factor=4):
        super(VideoSuperResolution, self).__init__()
        self.feature_extractor = MobileNetV3FeatureExtractor(pretrained=True)
        with torch.no_grad():
            sample_input = torch.randn(1, 3, 64, 64)
            features = self.feature_extractor(sample_input)
            feature_channels = features.shape[1]

        self.temporal_module = TemporalConsistencyModule(feature_channels)
        self.fusion = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.reconstruction = ReconstructionModule(feature_channels, scale_factor)
        self.prev_features = None

    def forward(self, x, scale_factor=4, reset_state=True):
        batch_size, seq_len, c, h, w = x.shape
        outputs = []

        if reset_state:
            self.prev_features = None

        for t in range(seq_len):
            current_frame = x[:, t, :, :, :]
            current_features = self.feature_extractor(current_frame)
            temporally_consistent_features = self.temporal_module(current_features, self.prev_features)
            self.prev_features = current_features.detach()

            fused_features = self.fusion(temporally_consistent_features)
            sr_frame = self.reconstruction(fused_features)
            outputs.append(sr_frame)

        return torch.stack(outputs, dim=1)
