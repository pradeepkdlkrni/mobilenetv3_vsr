# Fixed model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

FEATURE_CHANNELS = 64

class MobileNetV3FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV3FeatureExtractor, self).__init__()

        mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        self.features = nn.Sequential(*list(mobilenet.features.children())[:6])

    def forward(self, x):
        return self.features(x)

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

        # Dynamic residual block depth based on scale
        block_depth = 3 if scale_factor <= 2 else 5 if scale_factor <= 4 else 8
        self.res_blocks = nn.Sequential(*[ResidualBlock(FEATURE_CHANNELS) for _ in range(block_depth)])

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
        self.scale_factor = scale_factor

        self.feature_extractor = MobileNetV3FeatureExtractor(pretrained=True)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 64, 64)
            feature_channels = self.feature_extractor(dummy).shape[1]

        self.feature_refiner = nn.Sequential(
            ResidualBlock(feature_channels),
            ResidualBlock(feature_channels)
        )

        self.temporal_module = TemporalConsistencyModule(feature_channels)

        # Spatial and temporal fusion split
        self.spatial_fusion = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.temporal_fusion = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)

        self.reconstruction = ReconstructionModule(feature_channels, scale_factor)
        self.prev_features = None

    def forward(self, x, scale_factor=None, reset_state=True):
        if scale_factor is None:
            scale_factor = self.scale_factor

        batch_size, seq_len, c, h, w = x.shape
        outputs = []

        if reset_state:
            self.prev_features = None

        for t in range(seq_len):
            current_frame = x[:, t, :, :, :]
            current_features = self.feature_extractor(current_frame)
            current_features = self.feature_refiner(current_features)

            temporal_out = self.temporal_module(current_features, self.prev_features)
            if self.prev_features is not None:
                temporal_out = temporal_out + self.prev_features
            self.prev_features = temporal_out.detach()

            fused_features = self.spatial_fusion(current_features) + self.temporal_fusion(temporal_out)

            # Add residual skip from LR input
            upsampled_lr = F.interpolate(current_frame, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            sr_frame = self.reconstruction(fused_features)
            
            if sr_frame.shape != upsampled_lr.shape:
               upsampled_lr = F.interpolate(upsampled_lr, size=sr_frame.shape[2:], mode='bilinear', align_corners=False)
			
            sr_frame = sr_frame + upsampled_lr

            outputs.append(sr_frame)

        return torch.stack(outputs, dim=1)
