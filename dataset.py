import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
# Other imports...

# === BEGIN: Utility class to fix HR/LR scale mismatches ===
class ScaleAwarePairAligner:
    def __init__(self, scale_factor, method='hybrid', verbose=False):
        self.scale = scale_factor
        self.method = method
        self.verbose = verbose

    def __call__(self, hr_img, lr_img):
        hr_h, hr_w = hr_img.shape[:2]
        lr_h, lr_w = lr_img.shape[:2]

        expected_hr_h = lr_h * self.scale
        expected_hr_w = lr_w * self.scale

        if hr_h == expected_hr_h and hr_w == expected_hr_w:
            return hr_img, lr_img  # Perfect match

        #if self.verbose:
            ##print(f"[ScaleAlign] Mismatch for scale={self.scale}")
            ##print(f"  HR size: ({hr_h}, {hr_w}), LR*scale: ({expected_hr_h}, {expected_hr_w})")

        if self.method == 'crop' or (self.method == 'hybrid' and hr_h >= expected_hr_h and hr_w >= expected_hr_w):
            min_h = min(hr_h, expected_hr_h)
            min_w = min(hr_w, expected_hr_w)
            min_h = (min_h // self.scale) * self.scale
            min_w = (min_w // self.scale) * self.scale

            cropped_hr = hr_img[:min_h, :min_w]
            cropped_lr = lr_img[:min_h // self.scale, :min_w // self.scale]
            return cropped_hr, cropped_lr

        elif self.method in ['resize', 'hybrid']:
            new_hr_h = lr_h * self.scale
            new_hr_w = lr_w * self.scale

            resized_hr = cv2.resize(hr_img, (new_hr_w, new_hr_h), interpolation=cv2.INTER_CUBIC)
            regenerated_lr = cv2.resize(resized_hr, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
            return resized_hr, regenerated_lr

        if self.method == 'resize':
            resized_hr = cv2.resize(hr_img, (lr_w * self.scale, lr_h * self.scale), interpolation=cv2.INTER_CUBIC)
            regenerated_lr = cv2.resize(resized_hr, (lr_w, lr_h), interpolation=cv2.INTER_AREA)
            return resized_hr, regenerated_lr

        else:
            raise ValueError(f"Unknown method '{self.method}'. Use 'crop', 'resize', or 'hybrid'.")
# === END: Utility class ===

class VideoDataset(Dataset):
    def __init__(self, lr_folder, hr_folder, sequence_length=5, transform=None, scale_factor=4):
        self.lr_folder = lr_folder
        self.hr_folder = hr_folder
        self.sequence_length = sequence_length
        self.transform = transform
        self.scale_factor = scale_factor
        self.pair_aligner = ScaleAwarePairAligner(scale_factor, method='resize')
        
        # Get video sequence folders (00001_0003, etc.)
        self.sequence_folders = sorted([f for f in os.listdir(lr_folder) 
                                     if os.path.isdir(os.path.join(lr_folder, f))])
        
        if len(self.sequence_folders) == 0:
            raise ValueError(f"No subdirectories found in {lr_folder}. Check your path.")
            
        ##print(f"Found {len(self.sequence_folders)} sequence folders in {lr_folder}")
        ##print(f"Using scale factor: {scale_factor}x")
        
        # Create mapping for efficient retrieval
        self.valid_sequences = []
        
        for seq_folder in self.sequence_folders:
            lr_seq_path = os.path.join(lr_folder, seq_folder)
            hr_seq_path = os.path.join(hr_folder, seq_folder)
            
            # Ensure both LR and HR folders for this sequence exist
            if not os.path.exists(hr_seq_path):
                #print(f"Warning: HR folder {hr_seq_path} does not exist, skipping sequence {seq_folder}")
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
        
        #print(f"Total valid sequences: {len(self.valid_sequences)}")
        
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
            # Load HR frame first to determine dimensions
            hr_path = os.path.join(self.hr_folder, seq_folder, hr_frames_list[i])
            lr_path = os.path.join(self.lr_folder, seq_folder, lr_frames_list[i])
            
            # Load HR
            hr_frame = cv2.imread(hr_path)
            hr_frame = cv2.cvtColor(hr_frame, cv2.COLOR_BGR2RGB)
            
            hr_h, hr_w = hr_frame.shape[:2]
            #print(" hr_frame size just after reading from disk,  height = ", hr_h , "width=", hr_w)
            # Load LR
            lr_frame = cv2.imread(lr_path)
            lr_frame = cv2.cvtColor(lr_frame, cv2.COLOR_BGR2RGB)
            lr_h, lr_w = lr_frame.shape[:2]
            #print(" lr size just after reading from disk,  height = ", lr_h , "width=", lr_w)

            # Verify: Do they match scale?
            hr_frame, lr_frame = self.pair_aligner(hr_frame, lr_frame)

             

            # Verify dimensions match the expected scale factor relationship
            assert hr_frame.shape[0] == lr_frame.shape[0] * self.scale_factor
            assert hr_frame.shape[1] == lr_frame.shape[1] * self.scale_factor
            
            if self.transform:
                if isinstance(self.transform, tuple) and len(self.transform) >= 2:
                    lr_transform, hr_transform = self.transform[:2]
                    lr_frame = lr_transform(lr_frame)
                    hr_frame = hr_transform(hr_frame)
                elif callable(self.transform):
                    lr_frame = self.transform(lr_frame)
                    hr_frame = self.transform(hr_frame)
                else:
                    # Default conversion to Tensor
                    lr_frame = torch.from_numpy(lr_frame.transpose(2, 0, 1)).float() / 255.0
                    hr_frame = torch.from_numpy(hr_frame.transpose(2, 0, 1)).float() / 255.0
            else:
                # Default conversion to Tensor if no transform
                lr_frame = torch.from_numpy(lr_frame.transpose(2, 0, 1)).float() / 255.0
                hr_frame = torch.from_numpy(hr_frame.transpose(2, 0, 1)).float() / 255.0
                
            lr_sequence.append(lr_frame)
            hr_sequence.append(hr_frame)
        
        # Stack frames into tensors
        lr_sequence = torch.stack(lr_sequence)
        hr_sequence = torch.stack(hr_sequence)
        lr_paths = [os.path.join(self.lr_folder, seq_folder, fname) for fname in lr_frames_list]
        return lr_sequence, hr_sequence, lr_paths