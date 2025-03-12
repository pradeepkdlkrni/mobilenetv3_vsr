# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 15:00:09 2025

@author: prade
"""
import torch
import torch.nn as nn  
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class TemporalConsistencyLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super(TemporalConsistencyLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
    
    def forward(self, output_frames, target_frames, prev_output_frames=None):
        # Standard reconstruction loss
        recon_loss = self.mse(output_frames, target_frames)
        
        # If no previous output is available, return only reconstruction loss
        if prev_output_frames is None:
            return recon_loss
        
        # Compute temporal difference in outputs
        current_temp_diff = output_frames[:, 1:] - output_frames[:, :-1]
        
        # Compute temporal difference in targets
        target_temp_diff = target_frames[:, 1:] - target_frames[:, :-1]
        
        # Temporal consistency loss
        temp_loss = self.mse(current_temp_diff, target_temp_diff)
        
        # Combined loss
        return recon_loss + self.alpha * temp_loss

# Perceptual loss using VGG features
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features.to(device)
        self.vgg_layers = nn.Sequential(*list(vgg.children())[:16])
        
        # Freeze VGG parameters
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        
        self.mse = nn.MSELoss()
    
    def forward(self, output, target):
        output_features = self.vgg_layers(output.view(-1, 3, output.shape[-2], output.shape[-1]))
        target_features = self.vgg_layers(target.view(-1, 3, target.shape[-2], target.shape[-1]))
        
        return self.mse(output_features, target_features)

# Combined loss function
class CombinedLoss(nn.Module):
    def __init__(self, lambda_rec=1.0, lambda_perc=0.1, lambda_temp=0.8):
        #super(CombinedLoss, self).__init__()
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_perc = lambda_perc
        self.lambda_temp = lambda_temp
        
        self.rec_loss = nn.MSELoss()
        self.perc_loss = PerceptualLoss()
        self.temp_loss = TemporalConsistencyLoss()
    
    def forward(self, output, target, prev_output=None):
        # Reshape if needed for perceptual loss
        b, t, c, h, w = output.shape
        
        rec = self.rec_loss(output, target)
        perc = self.perc_loss(output.view(-1, c, h, w), target.view(-1, c, h, w))
        temp = self.temp_loss(output, target, prev_output)
        
        return self.lambda_rec * rec + self.lambda_perc * perc + self.lambda_temp * temp
