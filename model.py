# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 14:59:49 2025

@author: prade
"""

# File: model.py
# Contains all model-related classes

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from torch.nn.utils import weight_norm

# Define constants
FEATURE_CHANNELS = 64

# MobileNetV3 Feature Extractor
class MobileNetV3FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV3FeatureExtractor, self).__init__()
        # Load MobileNetV3 Small model pretrained on ImageNet
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Use only the features part (before classifier)
        self.features = nn.Sequential(*list(mobilenet.features.children())[:3])
        
        # Freeze the parameters
        if pretrained:
            for param in self.features.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        return self.features(x)

# Residual Block for the reconstruction network
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.conv2 = weight_norm(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return out

# Temporal Consistency Module
class TemporalConsistencyModule(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super(TemporalConsistencyModule, self).__init__()
        self.conv_h = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv_x = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv_o = nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1)
        
        # Optical flow estimation layers
        self.flow_conv1 = nn.Conv2d(in_channels * 2, 24, kernel_size=3, padding=1)
        self.flow_conv2 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.flow_conv3 = nn.Conv2d(24, 2, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, current, previous=None):
        if previous is None:
            return current
        
        # Estimate optical flow between frames
        flow_input = torch.cat([current, previous], dim=1)
        flow = self.relu(self.flow_conv1(flow_input))
        flow = self.relu(self.flow_conv2(flow))
        flow = self.flow_conv3(flow)
        
        # Warp previous features
        b, c, h, w = previous.size()
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid = torch.stack((grid_x, grid_y), dim=2).float().to(current.device)
        grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)
        flow_grid = grid + flow.permute(0, 2, 3, 1)
        
        # Normalize coordinates to [-1, 1]
        flow_grid[:, :, :, 0] = 2.0 * flow_grid[:, :, :, 0] / (w - 1) - 1.0
        flow_grid[:, :, :, 1] = 2.0 * flow_grid[:, :, :, 1] / (h - 1) - 1.0
        
        # Warp features using the flow field
        warped_previous = F.grid_sample(previous, flow_grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        # Temporal fusion
        h_state = self.conv_h(warped_previous)
        x_state = self.conv_x(current)
        combined = self.relu(h_state + x_state)
        out = self.conv_o(combined)
        
        return out

# Reconstruction module
# Reconstruction module
class ReconstructionModule(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(ReconstructionModule, self).__init__()
        self.scale_factor = scale_factor
        
        # Initial convolution to process features
        self.conv1 = nn.Conv2d(in_channels, FEATURE_CHANNELS, kernel_size=3, padding=1)
        
        # Residual blocks for feature processing
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(FEATURE_CHANNELS) for _ in range(5)]
        )
        
        # For upscaling from 8×8 to 128×128, we need 4x upscaling
        # We'll do three PixelShuffle operations: 8×8 -> 16×16 -> 32×32 -> 64×64 -> 128×128
        self.upscale = nn.Sequential(
            # First upscale: 8×8 -> 16×16
            nn.Conv2d(FEATURE_CHANNELS, FEATURE_CHANNELS * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            
            # Second upscale: 16×16 -> 32×32
            nn.Conv2d(FEATURE_CHANNELS, FEATURE_CHANNELS * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            
            # Third upscale: 32×32 -> 64×64
            nn.Conv2d(FEATURE_CHANNELS, FEATURE_CHANNELS * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            
            # Fourth upscale: 64×64 -> 128×128
            nn.Conv2d(FEATURE_CHANNELS, FEATURE_CHANNELS * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution to generate RGB output
        self.final_conv = nn.Conv2d(FEATURE_CHANNELS, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
       # print(f"Reconstruction input shape: {x.shape}")
        
        # Initial feature processing
        x = self.conv1(x)
        #print(f"After initial conv: {x.shape}")
        
        # Residual feature enhancement
        x = self.res_blocks(x)
        #print(f"After residual blocks: {x.shape}")
        
        # Upscaling
        x = self.upscale(x)
        #print(f"After upscaling: {x.shape}")
        
        # Final output generation
        x = self.final_conv(x)
        #print(f"Final output shape: {x.shape}")
        
        return x




# Complete Video Super Resolution Model
class VideoSuperResolution(nn.Module):
    def __init__(self, scale_factor=4):
        super(VideoSuperResolution, self).__init__()
        #print("Initializing MobileNetV3 Feature Extractor...")
        # Feature extractor
        self.feature_extractor = MobileNetV3FeatureExtractor(pretrained=True)
        #print("Feature Extractor initialized.")
        #print("Running a dummy forward pass through feature extractor...")
        # Determine the feature size
        with torch.no_grad():
            sample_input = torch.randn(1, 3, 64, 64)
            features = self.feature_extractor(sample_input)
            feature_channels = features.shape[1]
        #print(f"Feature extractor output shape: {features.shape}")
        
        # Temporal consistency module
        #print("Initializing Temporal Consistency Module...")
        self.temporal_module = TemporalConsistencyModule(feature_channels)
        #print("Temporal Consistency Module initialized.")
        
        # Feature fusion (additional convolutional layer to combine features)
        #print("Initializing Feature Fusion Layer...")
        self.fusion = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        #print("Feature Fusion Layer initialized.")
        
        # Reconstruction module
       # print("Initializing Reconstruction Module...")
        self.reconstruction = ReconstructionModule(feature_channels, scale_factor)
       # print("Reconstruction Module initialized.")
        
        # Store previous features for temporal consistency
        self.prev_features = None
       # print("Model initialization complete.")
    
    def forward(self, x, reset_state=False):
        
        batch_size, seq_len, c, h, w = x.shape
        print(f"Input shape before processing: {x.shape}")
        outputs = []
        
        if reset_state:
            self.prev_features = None
        
        # Process each frame in the sequence
        for t in range(seq_len):
            # Extract current frame
            current_frame = x[:, t, :, :, :]
            #print(f"Current frame shape: {current_frame.shape}")
            # Extract features using MobileNetV3
            current_features = self.feature_extractor(current_frame)
            #print(f"Features shape after extraction: {current_features.shape}")
            # Apply temporal consistency if we have previous features
            temporally_consistent_features = self.temporal_module(current_features, self.prev_features)
            ##print(f"Features shape after temporal module: {temporally_consistent_features.shape}")
            # Update previous features for next iteration
            self.prev_features = current_features.detach()
            
            # Fuse features
            fused_features = self.fusion(temporally_consistent_features)
            #print(f"Features shape after fusion: {fused_features.shape}")
            # Reconstruct high-resolution frame
            sr_frame = self.reconstruction(fused_features)
            print(f"SR frame shape after reconstruction: {sr_frame.shape}")
            outputs.append(sr_frame)
        
        # Stack outputs along sequence dimension
        return torch.stack(outputs, dim=1)