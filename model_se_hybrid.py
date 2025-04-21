


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

# Paste in your model_se_hybrid.py (top with other imports)
import math
import torch.nn.init as init

class TinyTransformerBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).reshape(b, 3, self.heads, c // self.heads, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(b, c, h, w)

        out = self.proj(out)
        out = self.norm(out + x)  # Residual
        return out



class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super().__init__()
        # Channel Attention
        self.channel_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        avg_out = self.channel_mlp(x)
        ch_att = self.sigmoid_channel(avg_out)
        x = x * ch_att

        # Spatial Attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        mean_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_input = torch.cat([max_pool, mean_pool], dim=1)
        sp_att = self.spatial(spatial_input)

        return x * sp_att

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, state):
        h_cur, c_cur = state
        combined = torch.cat([x, h_cur], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, conv_output.shape[1] // 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, (h_next, c_next)
    
class SEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        reduction = max(1, channels // 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w  # residual connection for attention

# Fixed model.py
FEATURE_CHANNELS = 64

class MobileNetV3FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV3FeatureExtractor, self).__init__()

        mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
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


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.reset_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h_prev):
        if h_prev is None:
            h_prev = torch.zeros_like(x)
        combined = torch.cat([x, h_prev], dim=1)
        z = self.sigmoid(self.update_gate(combined))
        r = self.sigmoid(self.reset_gate(combined))
        h_candidate = self.tanh(self.out_gate(torch.cat([x, r * h_prev], dim=1)))
        h_new = (1 - z) * h_prev + z * h_candidate
        return h_new


class VideoSuperResolution(nn.Module):
    def __init__(self, scale_factor=4):
        super(VideoSuperResolution, self).__init__()
        self.scale_factor = scale_factor
        self.lr_fusion = nn.Sequential(
            nn.Conv2d(6, self.scale_factor * 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(self.scale_factor * 3),
            nn.Conv2d(self.scale_factor * 3, 3, kernel_size=3, padding=1)
        )

        self.feature_extractor = MobileNetV3FeatureExtractor(pretrained=True)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 64, 64)
            feature_channels = self.feature_extractor(dummy).shape[1]
        # Use the actual number of channels dynamically
        self.temporal_transformer = TinyTransformerBlock(feature_channels)
        self.feature_refiner = nn.Sequential(
            ResidualBlock(feature_channels),
            ResidualBlock(feature_channels)
        )

        #self.temporal_gru = ConvGRUCell(feature_channels, feature_channels)
        self.temporal_lstm = ConvLSTMCell(feature_channels, feature_channels)
        self.temporal_state = None

        # Spatial and temporal fusion split
        self.spatial_fusion = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.temporal_fusion = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)

        
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=5, padding=2)
        )
        self.reconstruction = nn.Sequential(
            *[nn.Sequential(ResidualBlock(feature_channels), CBAMBlock(feature_channels)) for _ in range(4)],
            *[ResidualBlock(feature_channels) for _ in range(4)],
            nn.Conv2d(feature_channels, 3 * scale_factor * scale_factor, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )

        self.lr_fusion = nn.Sequential(
            nn.Conv2d(6, FEATURE_CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(FEATURE_CHANNELS),
            nn.Conv2d(FEATURE_CHANNELS, 3, kernel_size=3, padding=1)
        )
        nn.init.zeros_(self.lr_fusion[-1].weight)
        nn.init.zeros_(self.lr_fusion[-1].bias)

        self.prev_features = None

    def forward(self, x, scale_factor=None, reset_state=True):
        if scale_factor is None:
            scale_factor = self.scale_factor

        batch_size, seq_len, c, h, w = x.shape
        outputs = []

        if reset_state:
            self.prev_features = None
            self.temporal_state = None

        for t in range(seq_len):
            current_frame = x[:, t, :, :, :]
            current_features = self.feature_extractor(current_frame)
            current_features = self.feature_refiner(current_features)

            #if self.prev_features is None:
                #self.prev_features = torch.zeros_like(current_features)
            #temporal_out = self.temporal_gru(current_features, self.prev_features)
            
            if self.temporal_state is None:
                h_cur = torch.zeros(current_features.size(0), current_features.size(1), current_features.size(2), current_features.size(3), device=current_features.device)
                c_cur = torch.zeros_like(h_cur)
                self.temporal_state = (h_cur, c_cur)

            temporal_out, (h_next, c_next) = self.temporal_lstm(current_features, self.temporal_state)
            self.temporal_state = (h_next.detach(), c_next.detach())
            
            self.prev_features = temporal_out.detach()
            if self.prev_features is not None:
                temporal_out = temporal_out + self.prev_features
            self.prev_features = temporal_out.detach()

            fused_temporal = self.temporal_fusion(temporal_out)
            fused_features = self.spatial_fusion(current_features) + fused_temporal
            fused_features = self.temporal_transformer(fused_features)

            # Add residual skip from LR input
            upsampled_lr = F.interpolate(current_frame, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            sr_frame = self.reconstruction(fused_features)
            
            if sr_frame.shape != upsampled_lr.shape:
               upsampled_lr = F.interpolate(upsampled_lr, size=sr_frame.shape[2:], mode='bilinear', align_corners=False)
			
            concat = torch.cat([sr_frame, upsampled_lr], dim=1)
            sr_frame = self.lr_fusion(concat)

            
            concat = torch.cat([sr_frame, upsampled_lr], dim=1)
            alpha = min(1.0, getattr(self, 'current_epoch', 1) / 3.0)
            sr_learned = self.lr_fusion(concat)
            sr_frame = alpha * sr_learned + (1 - alpha) * (sr_frame + upsampled_lr)
            sr_frame = sr_frame + upsampled_lr
            outputs.append(sr_frame)
    

        return torch.stack(outputs, dim=1)
