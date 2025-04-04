import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, lr_folder, hr_folder, sequence_length=5, transform=None):
        self.lr_folder = lr_folder
        self.hr_folder = hr_folder
        self.sequence_length = sequence_length
        self.transform = transform
        
        # For Vimeo septuplet structure: 
        # F://vimeo_septuplet_full//Arranged//train//lr//00001_0003//frame_00001.png

        # Get video sequence folders (00001_0003, etc.)
        self.sequence_folders = sorted([f for f in os.listdir(lr_folder) 
                                     if os.path.isdir(os.path.join(lr_folder, f))])
        
        if len(self.sequence_folders) == 0:
            raise ValueError(f"No subdirectories found in {lr_folder}. Check your path.")
            
        print(f"Found {len(self.sequence_folders)} sequence folders in {lr_folder}")
        
        # Create mapping for efficient retrieval
        self.valid_sequences = []
        
        for seq_folder in self.sequence_folders:
            lr_seq_path = os.path.join(lr_folder, seq_folder)
            hr_seq_path = os.path.join(hr_folder, seq_folder)
            
            # Ensure both LR and HR folders for this sequence exist
            if not os.path.exists(hr_seq_path):
                print(f"Warning: HR folder {hr_seq_path} does not exist, skipping sequence {seq_folder}")
                continue
                
            # Get frames in this sequence
            lr_frames = sorted([f for f in os.listdir(lr_seq_path) if f.endswith('.png') or f.endswith('.jpg')])
            hr_frames = sorted([f for f in os.listdir(hr_seq_path) if f.endswith('.png') or f.endswith('.jpg')])
            
            # Check if there are enough frames for a sequence
            if len(lr_frames) < sequence_length or len(hr_frames) < sequence_length:
                continue
                
            # Store valid sequence
            self.valid_sequences.append({
                'folder': seq_folder,
                'lr_frames': lr_frames[:sequence_length],  # Use first sequence_length frames
                'hr_frames': hr_frames[:sequence_length]
            })
        
        print(f"Total valid sequences: {len(self.valid_sequences)}")
        
        if len(self.valid_sequences) == 0:
            raise ValueError(f"No valid sequences found that contain {sequence_length} frames in both LR and HR.")

    def __len__(self):
        return len(self.valid_sequences)

    def __getitem__(self, idx):
        sequence = self.valid_sequences[idx]
        seq_folder = sequence['folder']
        lr_frames_list = sequence['lr_frames']
        hr_frames_list = sequence['hr_frames']
        
        # Load sequence of frames
        lr_sequence = []
        hr_sequence = []
        
        for i in range(self.sequence_length):
            # Load LR frame
            lr_path = os.path.join(self.lr_folder, seq_folder, lr_frames_list[i])
            lr_frame = cv2.imread(lr_path)
            if lr_frame is None:
                raise ValueError(f"Could not read image at {lr_path}")
            lr_frame = cv2.cvtColor(lr_frame, cv2.COLOR_BGR2RGB)
            
            # Load HR frame
            hr_path = os.path.join(self.hr_folder, seq_folder, hr_frames_list[i])
            hr_frame = cv2.imread(hr_path)
            if hr_frame is None:
                raise ValueError(f"Could not read image at {hr_path}")
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
            
            lr_sequence.append(lr_frame)
            hr_sequence.append(hr_frame)
        
        # Stack frames into tensors
        lr_sequence = torch.stack(lr_sequence)
        hr_sequence = torch.stack(hr_sequence)
        
        return lr_sequence, hr_sequence