# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 15:01:00 2025

@author: prade
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm

import torch
NUM_EPOCHS = 10

def train(model, train_loader, criterion, optimizer, epoch, device, scale_factor):
    model.train()
    running_loss = 0.0
    prev_output = None
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}") as pbar:
        for i, (lr_frames, hr_frames) in enumerate(pbar):
            # Move data to device
            lr_frames = lr_frames.to(device)
            hr_frames = hr_frames.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Reset state at the beginning of a batch
            #output = model(lr_frames, reset_state=True)
            output = model(lr_frames, scale_factor=scale_factor, reset_state=True)
            #print(f"Output shape: {output.shape}, Target shape: {hr_frames.shape}")
            # Compute loss with temporal consistency
            print(f"[DEBUG] output: {output.shape}, hr: {hr_frames.shape}")
            loss = criterion(output, hr_frames, prev_output)
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            
            # Update prev_output for next iteration
            prev_output = output.detach()
            
            # Update statistics
            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (i + 1))
    
    return running_loss / len(train_loader)

# Validation function
def validate(model, val_loader, criterion, device, scale_factor):
    model.eval()
    val_loss = 0.0
    psnr_total = 0.0
    prev_output = None
    
    with torch.no_grad():
        for lr_frames, hr_frames in val_loader:
            lr_frames = lr_frames.to(device)
            hr_frames = hr_frames.to(device)
            
            # Reset state at the beginning of a batch
            #output = model(lr_frames, reset_state=True)
            output = model(lr_frames, scale_factor=scale_factor, reset_state=True)
            
            target_size = hr_frames.shape[-2:]  # (H, W)
            B, T, C, H, W = output.shape
            output = output.view(B * T, C, H, W)

            # Target size from hr_frames
            target_H, target_W = hr_frames.shape[-2:]
            output = F.interpolate(output, size=(target_H, target_W), mode='bilinear', align_corners=False)
            
            # Reshape back to [B, T, C, H, W]
            output = output.view(B, T, C, target_H, target_W) 
            print(f"[DEBUG] output: {output.shape}, hr: {hr_frames.shape}")
            # Compute loss
            loss = criterion(output, hr_frames, prev_output)
            val_loss += loss.item()
            
            # Update prev_output for next iteration
            prev_output = output.detach()
            
            # Compute PSNR
            mse = F.mse_loss(output, hr_frames)
            psnr = 10 * torch.log10(1.0 / mse)
            psnr_total += psnr.item()
    
    avg_loss = val_loss / len(val_loader)
    avg_psnr = psnr_total / len(val_loader)
    
    return avg_loss, avg_psnr

# SSIM calculation function
def calculate_ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def resize_to_match(input_tensor, target_tensor):
    return F.interpolate(input_tensor.unsqueeze(0), size=target_tensor.shape[-2:], mode='bilinear', align_corners=False).squeeze(0)


# Test function
def test(model, test_loader, device, scale_factor, save_path='results'):
    model.eval()
    os.makedirs(save_path, exist_ok=True)
    
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for batch_idx, (lr_frames, hr_frames) in enumerate(test_loader):
            lr_frames = lr_frames.to(device)
            hr_frames = hr_frames.to(device)
            
            # Reset state at the beginning of a batch
            #output = model(lr_frames, reset_state=True)
            output = model(lr_frames, scale_factor=scale_factor, reset_state=True)
            # Save some results for visualization
            if batch_idx < 5:  # Save first 5 batches
                for seq_idx in range(min(2, output.size(0))):  # Save first 2 sequences in each batch
                    sequence_dir = os.path.join(save_path, f'batch_{batch_idx}_seq_{seq_idx}')
                    os.makedirs(sequence_dir, exist_ok=True)
                    
                    for frame_idx in range(output.size(1)):
                        # Get frames
                        lr = lr_frames[seq_idx, frame_idx].cpu()
                        sr = output[seq_idx, frame_idx].cpu()
                        hr = hr_frames[seq_idx, frame_idx].cpu()
                        
                        # Convert to numpy and denormalize
                        lr_np = lr.permute(1, 2, 0).numpy() * 255
                        sr_np = sr.permute(1, 2, 0).numpy() * 255
                        hr_np = hr.permute(1, 2, 0).numpy() * 255
                        
                        # Ensure range is correct
                        lr_np = np.clip(lr_np, 0, 255).astype(np.uint8)
                        sr_np = np.clip(sr_np, 0, 255).astype(np.uint8)
                        hr_np = np.clip(hr_np, 0, 255).astype(np.uint8)
                        
                        # Save images
                        cv2.imwrite(os.path.join(sequence_dir, f'frame_{frame_idx}_lr.png'), 
                                   cv2.cvtColor(lr_np, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(os.path.join(sequence_dir, f'frame_{frame_idx}_sr.png'), 
                                   cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(os.path.join(sequence_dir, f'frame_{frame_idx}_hr.png'), 
                                   cv2.cvtColor(hr_np, cv2.COLOR_RGB2BGR))
            
            # Calculate metrics
            for seq_idx in range(output.size(0)):
                for frame_idx in range(output.size(1)):
                    sr = output[seq_idx, frame_idx].cpu()
                    hr = hr_frames[seq_idx, frame_idx].cpu()
            
                    # Ensure output resolution matches HR
                    if sr.shape != hr.shape:
                        sr = resize_to_match(sr, hr)
            
                    # Calculate PSNR
                    mse = F.mse_loss(sr, hr).item()
                    psnr = 10 * np.log10(1.0 / mse if mse > 0 else 1e-8)
                    psnr_values.append(psnr)
            
                    # Calculate SSIM
                    sr_np = sr.permute(1, 2, 0).numpy() * 255
                    hr_np = hr.permute(1, 2, 0).numpy() * 255
                    sr_np = np.clip(sr_np, 0, 255).astype(np.uint8)
                    hr_np = np.clip(hr_np, 0, 255).astype(np.uint8)
            
                    # Convert to grayscale for SSIM
                    sr_gray = cv2.cvtColor(sr_np, cv2.COLOR_RGB2GRAY)
                    hr_gray = cv2.cvtColor(hr_np, cv2.COLOR_RGB2GRAY)
            
                    # Calculate SSIM
                    ssim = calculate_ssim(sr_gray, hr_gray)
                    ssim_values.append(ssim)
    
    # Calculate average metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    return avg_psnr, avg_ssim

# SSIM calculation function
def calculate_ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()