import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import vgg16, VGG16_Weights

def compute_motion_weight(current_frame, prev_frame):
    # Absolute difference in luminance space
    diff = torch.mean(torch.abs(current_frame - prev_frame), dim=1, keepdim=True)
    norm_diff = diff / (diff.max() + 1e-6)

    return norm_diff + 1.0  # Ensure weights are ≥1.0


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, prediction, target):
        return torch.mean(torch.sqrt((prediction - target) ** 2 + self.eps))

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.charbonnier = CharbonnierLoss()

    def forward(self, output, target):
        if output.shape != target.shape:
            b, t, c, h, w = output.shape
            _, _, _, target_h, target_w = target.shape
            if h != target_h or w != target_w:
                output_reshaped = output.view(-1, c, h, w)
                output_resized = F.interpolate(output_reshaped, size=(target_h, target_w), mode='bilinear', align_corners=False)
                output = output_resized.view(b, t, c, target_h, target_w)

        return self.charbonnier(output, target)
    
'''
class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.mse_loss = nn.L1Loss()
        
    def forward(self, output, target):
        """
        Calculate reconstruction loss
        
        Args:
            output (torch.Tensor): Shape [batch_size, sequence_length, channels, height, width]
            target (torch.Tensor): Shape [batch_size, sequence_length, channels, height, width]
        """
        # Ensure shapes match before computing loss
        if output.shape != target.shape:
            # Resize output to match target if necessary
            b, t, c, h, w = output.shape
            target_b, target_t, target_c, target_h, target_w = target.shape
            
            if h != target_h or w != target_w:
                output_reshaped = output.view(-1, c, h, w)
                output_resized = F.interpolate(output_reshaped, size=(target_h, target_w), mode='bilinear', align_corners=False)
                output = output_resized.view(b, t, c, target_h, target_w)
                
        return self.mse_loss(output, target)
'''
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:16])
        
        # Freeze the VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def forward(self, output, target):
        """
        Calculate perceptual loss using VGG16 features
        
        Args:
            output (torch.Tensor): Shape [batch_size*sequence_length, channels, height, width]
            target (torch.Tensor): Shape [batch_size*sequence_length, channels, height, width]
        """
        # Ensure output and target have the same spatial dimensions for VGG
        if output.shape[2:] != target.shape[2:]:
            output = F.interpolate(output, size=target.shape[2:], mode='bilinear', align_corners=False)
            
        output_features = self.feature_extractor(output)
        target_features = self.feature_extractor(target)
        
        return F.l1_loss(output_features, target_features)

class TemporalConsistencyLoss(nn.Module):
    def __init__(self):
        super(TemporalConsistencyLoss, self).__init__()
        
    def forward(self, output, target, prev_output=None):
        """
        Calculate temporal consistency loss
        
        Args:
            output (torch.Tensor): Shape [batch_size, sequence_length, channels, height, width]
            target (torch.Tensor): Shape [batch_size, sequence_length, channels, height, width]
            prev_output (torch.Tensor, optional): Previous output for temporal consistency
        """
        if prev_output is None or output.shape[1] <= 1:
            return torch.tensor(0.0, device=output.device)
            
        # Compute temporal differences
        curr_frames = output[:, 1:]  # Exclude first frame
        prev_frames = output[:, :-1]  # Exclude last frame
        
        # Ensure shapes match before computing loss
        if curr_frames.shape[3:] != prev_frames.shape[3:]:
            b, t, c, h, w = curr_frames.shape
            prev_h, prev_w = prev_frames.shape[3:]
            
            if h != prev_h or w != prev_w:
                curr_frames_reshaped = curr_frames.contiguous().view(-1, c, h, w)
                curr_frames_resized = F.interpolate(curr_frames_reshaped, size=(prev_h, prev_w), mode='bilinear', align_corners=False)
                curr_frames = curr_frames_resized.view(b, t, c, prev_h, prev_w)
        
        # Temporal smoothness loss
        temporal_diff = F.mse_loss(curr_frames, prev_frames)
        
        return temporal_diff

class CombinedLoss(nn.Module):
    def __init__(self, rec_weight=1.0, perc_weight=0.1, temp_weight=0.05):
        super(CombinedLoss, self).__init__()
        self.rec_loss = ReconstructionLoss()
        self.perc_loss = PerceptualLoss()
        self.temp_loss = TemporalConsistencyLoss()
        
        self.rec_weight = rec_weight
        self.perc_weight = perc_weight
        self.temp_weight = temp_weight
        
    def forward(self, output, target, prev_output=None):
        b, t, c, h, w = output.shape

        # Resize if needed
        if output.shape[3:] != target.shape[3:]:
            output = F.interpolate(output.view(-1, c, h, w), size=target.shape[3:], mode='bilinear', align_corners=False)
            output = output.view(b, t, c, target.shape[3], target.shape[4])

        rec_loss = 0.0
        temp_loss = 0.0
        perc_loss = 0.0

        for i in range(t):
            out_frame = output[:, i]
            tgt_frame = target[:, i]

            if i > 0:
                motion_weight = compute_motion_weight(tgt_frame, target[:, i - 1])
            else:
                motion_weight = torch.ones_like(out_frame[:, :1])  # no motion for first frame

            # === Weighted Reconstruction Loss ===
            rec = self.rec_loss(out_frame, tgt_frame)
            rec = (motion_weight * rec).mean()
            rec_loss += rec

            # === Perceptual Loss ===
            if self.perc_weight > 0:
                out_feat = self.perc_loss.feature_extractor(out_frame)
                tgt_feat = self.perc_loss.feature_extractor(tgt_frame)
                perc = F.l1_loss(out_feat, tgt_feat)
                perc_loss += perc

            # === Temporal Loss ===
            if self.temp_weight > 0 and i > 0:
                temp_loss += self.temp_loss(out_frame, tgt_frame, prev_output=output[:, i - 1])

        rec_loss /= t
        perc_loss /= max(1, t - 1)
        temp_loss /= max(1, t - 1)

        total_loss = self.rec_weight * rec_loss + self.perc_weight * perc_loss + self.temp_weight * temp_loss
        return total_loss
