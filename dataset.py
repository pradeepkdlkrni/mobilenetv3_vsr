# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 15:00:01 2025

@author: prade
"""

import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 15:00:01 2025

@author: prade
"""

class VideoDataset(Dataset):
    def __init__(self, lr_folder, hr_folder, sequence_length=5, transform=None):
        self.lr_folder = lr_folder
        self.hr_folder = hr_folder
        self.sequence_length = sequence_length
        self.transform = transform

        # Get sorted list of all LR and HR images
        # Check if the folder contains subdirectories
        if any(os.path.isdir(os.path.join(lr_folder, f)) for f in os.listdir(lr_folder)):
            # Handle directory structure with video subdirectories
            self.use_subdirs = True
            self.video_folders = sorted([f for f in os.listdir(lr_folder) if os.path.isdir(os.path.join(lr_folder, f))])
            
            # Create index mapping for efficient retrieval
            self.frame_indices = []
            for video_idx, video_name in enumerate(self.video_folders):
                lr_video_path = os.path.join(lr_folder, video_name)
                num_frames = len([f for f in os.listdir(lr_video_path) if f.endswith('.png')])
                
                # For each valid sequence starting point
                for start_idx in range(num_frames - sequence_length + 1):
                    self.frame_indices.append((video_idx, start_idx))
            
            #print(f"Found {len(self.video_folders)} video folders")
            #print(f"Total sequences available: {len(self.frame_indices)}")
        else:
            # Handle flat structure with all frames in one directory
            self.use_subdirs = False
            self.lr_frames = sorted([f for f in os.listdir(lr_folder) if f.endswith('.png')])
            self.hr_frames = sorted([f for f in os.listdir(hr_folder) if f.endswith('.png')])

            #print(f"Found {len(self.lr_frames)} LR frames in {lr_folder}")
            #print(f"Found {len(self.hr_frames)} HR frames in {hr_folder}")

            # Ensure there are enough frames for sequence extraction
            self.frame_indices = []
            num_frames = len(self.lr_frames)

            if num_frames < self.sequence_length:
                print(f"Error: Not enough frames in {lr_folder}. Found {num_frames}, need {self.sequence_length}.")
            else:
                for start_idx in range(num_frames - sequence_length + 1):
                    self.frame_indices.append(start_idx)

            print(f"Total sequences available: {len(self.frame_indices)}")

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        if self.use_subdirs:
            # Handle directory structure with video subdirectories
            video_idx, start_idx = self.frame_indices[idx]
            video_name = self.video_folders[video_idx]
            
            lr_frames = []
            hr_frames = []
            
            # Load sequence of frames
            for i in range(self.sequence_length):
                frame_idx = start_idx + i
                if len(self.hr_frames) == 0:
                    print(f"Warning: No HR frames found in {self.hr_folder}. Using placeholder HR frames.")
                    hr_frame = torch.zeros((3, 64, 64))  # Placeholder tensor
                else:
                    hr_path = os.path.join(self.hr_folder, self.hr_frames[frame_idx])
                    hr_frame = cv2.imread(hr_path)
                    hr_frame = cv2.cvtColor(hr_frame, cv2.COLOR_BGR2RGB)
                # Load LR frame
                lr_path = os.path.join(self.lr_folder, video_name, f"{frame_idx:05d}.png")
                lr_frame = cv2.imread(lr_path)
                lr_frame = cv2.cvtColor(lr_frame, cv2.COLOR_BGR2RGB)
                
                # Load HR frame
               # hr_path = os.path.join(self.hr_folder, video_name, f"{frame_idx:05d}.png")
               # hr_frame = cv2.imread(hr_path)
               # hr_frame = cv2.cvtColor(hr_frame, cv2.COLOR_BGR2RGB)
                
                # Apply transforms
                if self.transform:
                    # Check if transform is a tuple of (lr_transform, hr_transform)
                    if isinstance(self.transform, tuple) and len(self.transform) == 2:
                        lr_transform, hr_transform = self.transform
                        lr_frame = lr_transform(lr_frame)
                        hr_frame = hr_transform(hr_frame)
                    else:
                        # Apply the same transform to both
                        lr_frame = self.transform(lr_frame)
                        hr_frame = self.transform(hr_frame)
                else:
                    # Convert to tensor and normalize to [0, 1]
                    lr_frame = torch.from_numpy(lr_frame.transpose(2, 0, 1)).float() / 255.0
                    hr_frame = torch.from_numpy(hr_frame.transpose(2, 0, 1)).float() / 255.0
                
                lr_frames.append(lr_frame)
                hr_frames.append(hr_frame)
        else:
            # Handle flat structure with all frames in one directory
            start_idx = self.frame_indices[idx]
            
            # Load sequence of frames
            lr_frames = []
            hr_frames = []
            
            for i in range(self.sequence_length):
                frame_idx = start_idx + i
                lr_path = os.path.join(self.lr_folder, self.lr_frames[frame_idx])
                hr_path = os.path.join(self.hr_folder, self.hr_frames[frame_idx])
                
                lr_frame = cv2.imread(lr_path)
                lr_frame = cv2.cvtColor(lr_frame, cv2.COLOR_BGR2RGB)
                
                hr_frame = cv2.imread(hr_path)
                hr_frame = cv2.cvtColor(hr_frame, cv2.COLOR_BGR2RGB)
                
                # Apply transforms
                if self.transform:
                    # Check if transform is a tuple of (lr_transform, hr_transform)
                    if isinstance(self.transform, tuple) and len(self.transform) == 2:
                        lr_transform, hr_transform = self.transform
                        lr_frame = lr_transform(lr_frame)
                        hr_frame = hr_transform(hr_frame)
                    else:
                        # Apply the same transform to both
                        lr_frame = self.transform(lr_frame)
                        hr_frame = self.transform(hr_frame)
                else:
                    # Convert to tensor and normalize to [0, 1]
                    lr_frame = torch.from_numpy(lr_frame.transpose(2, 0, 1)).float() / 255.0
                    hr_frame = torch.from_numpy(hr_frame.transpose(2, 0, 1)).float() / 255.0
                
                lr_frames.append(lr_frame)
                hr_frames.append(hr_frame)
        
        # Stack frames into tensors
        lr_sequence = torch.stack(lr_frames)
        hr_sequence = torch.stack(hr_frames)
        
        return lr_sequence, hr_sequence