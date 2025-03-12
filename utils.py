# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 15:01:06 2025

@author: prade
"""
import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
SEQUENCE_LENGTH = 5
# Custom transforms
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(64),  # Random crop for data augmentation
    transforms.ToTensor()
])

valid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(64),  # Center crop for validation
    transforms.ToTensor()
])

class VideoDataset(Dataset):
    def __init__(self, lr_folder, hr_folder, sequence_length=SEQUENCE_LENGTH, transform=None):
        self.lr_folder = lr_folder
        self.hr_folder = hr_folder
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Get list of video folders
        self.video_folders = sorted([f for f in os.listdir(lr_folder) if os.path.isdir(os.path.join(lr_folder, f))])
        
        # Create index mapping for efficient retrieval
        self.frame_indices = []
        for video_idx, video_name in enumerate(self.video_folders):
            lr_video_path = os.path.join(lr_folder, video_name)
            num_frames = len([f for f in os.listdir(lr_video_path) if f.endswith('.png')])
            
            # For each valid sequence starting point
            for start_idx in range(num_frames - sequence_length + 1):
                self.frame_indices.append((video_idx, start_idx))
    
    def __len__(self):
        return len(self.frame_indices)
    
    def __getitem__(self, idx):
        video_idx, start_idx = self.frame_indices[idx]
        video_name = self.video_folders[video_idx]
        
        lr_frames = []
        hr_frames = []
        
        # Load sequence of frames
        for i in range(self.sequence_length):
            frame_idx = start_idx + i
            
            # Load LR frame
            lr_path = os.path.join(self.lr_folder, video_name, f"{frame_idx:05d}.png")
            lr_frame = cv2.imread(lr_path)
            lr_frame = cv2.cvtColor(lr_frame, cv2.COLOR_BGR2RGB)
            
            # Load HR frame
            hr_path = os.path.join(self.hr_folder, video_name, f"{frame_idx:05d}.png")
            hr_frame = cv2.imread(hr_path)
            hr_frame = cv2.cvtColor(hr_frame, cv2.COLOR_BGR2RGB)
            
            # Apply transforms if any
            if self.transform:
               lr_frame = self.transform[0](lr_frame)  # LR transform
               hr_frame = self.transform[1](hr_frame)  # HR transform
            else:
                # Convert to tensor and normalize to [0, 1]
                lr_frame = torch.from_numpy(lr_frame.transpose(2, 0, 1)).float() / 255.0
                hr_frame = torch.from_numpy(hr_frame.transpose(2, 0, 1)).float() / 255.0
            
            lr_frames.append(lr_frame)
            hr_frames.append(hr_frame)
        
        # Stack frames into tensors
        lr_sequence = torch.stack(lr_frames)
        hr_sequence = torch.stack(hr_frames)
        
        # Reshape to [sequence_length, channels, height, width]
        return lr_sequence, hr_sequence

# Define separate transforms for LR and HR
train_transform_lr = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),  # LR image size
    transforms.ToTensor()
])

train_transform_hr = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),  # HR image size (4x larger)
    transforms.ToTensor()
])

valid_transform_lr = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),  # LR validation size
    transforms.ToTensor()
])

valid_transform_hr = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),  # HR validation size
    transforms.ToTensor()
])

# For backward compatibility with your existing code
train_transform = (train_transform_lr, train_transform_hr)
valid_transform = (valid_transform_lr, valid_transform_hr)