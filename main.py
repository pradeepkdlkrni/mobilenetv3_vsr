# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 14:59:43 2025

@author: prade
"""
#%% Cell 1
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.optim as optim
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from os import chdir, getcwd

#%% Cell 2
# Import model components
from model import VideoSuperResolution
from dataset import VideoDataset
from loss import CombinedLoss
from train import train, validate, test
from utils import train_transform, valid_transform

#%% Cell 2
# Import model components




from utils import train_transform_lr, train_transform_hr, valid_transform_lr, valid_transform_hr
#%% Cell 3
# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
#%% Cell 4
# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
#%% Cell 5
# Define hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
SCALE_FACTOR = 32  # Super-resolution scale factor

#32 means 4 x upscaling.


SEQUENCE_LENGTH = 5  # Number of consecutive frames to process
#%%
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Define path to save SR images
os.makedirs('results', exist_ok=True)

def visualize_sr_results(model, test_loader, device, num_videos=3):
    """Visualizes and saves SR frames from the test set."""
    model.eval()
    with torch.no_grad():
        for vid_idx, (lr_frames, hr_frames) in enumerate(test_loader):
            lr_frames = lr_frames.to(device)
            sr_frames = model(lr_frames)  # Super-resolution

            # Convert tensors to NumPy images
            for frame_idx in range(lr_frames.shape[1]):  # Iterate over frames
                lr_image = lr_frames[0, frame_idx].cpu().numpy().transpose(1, 2, 0)
                sr_image = sr_frames[0, frame_idx].cpu().numpy().transpose(1, 2, 0)
                hr_image = hr_frames[0, frame_idx].cpu().numpy().transpose(1, 2, 0)

                # Normalize images to 0-255
                lr_image = (lr_image * 255.0).astype(np.uint8)
                sr_image = (sr_image * 255.0).astype(np.uint8)
                hr_image = (hr_image * 255.0).astype(np.uint8)

                # Save images
                cv2.imwrite(f"results/lr_video{vid_idx}_frame{frame_idx}.png", cv2.cvtColor(lr_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"results/sr_video{vid_idx}_frame{frame_idx}.png", cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"results/hr_video{vid_idx}_frame{frame_idx}.png", cv2.cvtColor(hr_image, cv2.COLOR_RGB2BGR))

                # Display first few frames
                if vid_idx < num_videos and frame_idx < 3:
                    plt.figure(figsize=(10, 4))
                    plt.subplot(1, 3, 1)
                    plt.imshow(lr_image)
                    plt.title("Low-Res")
                    plt.axis("off")

                    plt.subplot(1, 3, 2)
                    plt.imshow(sr_image)
                    plt.title("Super-Res")
                    plt.axis("off")

                    plt.subplot(1, 3, 3)
                    plt.imshow(hr_image)
                    plt.title("High-Res (GT)")
                    plt.axis("off")

                    plt.show()
            
            if vid_idx == num_videos - 1:  # Stop after visualizing a few videos
                break


#%% Cell 7
def main():
    # Create model
    #print("invoking model---- ")
    model = VideoSuperResolution(scale_factor=SCALE_FACTOR).to(device)
    #print("model ==", model)
    # Define loss function
    criterion = CombinedLoss().to(device)

    # Define optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Define dataset paths
    base_path = "F://vimeo_septuplet_full//Arranged_small//"
    train_lr_path = os.path.join(base_path, "train", "lr")
    train_hr_path = os.path.join(base_path, "train", "hr")
    val_lr_path = os.path.join(base_path, "val", "lr")
    val_hr_path = os.path.join(base_path, "val", "hr")
    test_lr_path = os.path.join(base_path, "test", "lr")
    test_hr_path = os.path.join(base_path, "test", "hr")
    
    # Verify paths exist
    for path in [train_lr_path, train_hr_path, val_lr_path, val_hr_path, test_lr_path, test_hr_path]:
        if not os.path.exists(path):
            print(f"Warning: Path does not exist: {path}")
    
    # Create datasets and dataloaders
    train_dataset = VideoDataset(
        lr_folder=train_lr_path,
        hr_folder=train_hr_path,
        sequence_length=SEQUENCE_LENGTH,
        transform=(train_transform_lr, train_transform_hr,SCALE_FACTOR)
    )
    
    val_dataset = VideoDataset(
        lr_folder=val_lr_path,
        hr_folder=val_hr_path,
        sequence_length=SEQUENCE_LENGTH,
        transform=(valid_transform_lr, valid_transform_hr,SCALE_FACTOR)
    )
    
    test_dataset = VideoDataset(
        lr_folder=test_lr_path,
        hr_folder=test_hr_path,
        sequence_length=SEQUENCE_LENGTH,
        transform=(valid_transform_lr, valid_transform_hr,SCALE_FACTOR)
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Check sample batch shapes
    for lr_frames, hr_frames in train_loader:
        print(f"LR input shape: {lr_frames.shape}")  # Expected: [batch_size, sequence_length, channels, height, width]
        print(f"HR target shape: {hr_frames.shape}")
        break  # Just print the first batch
    
    # Training log
    train_losses = []
    val_losses = []
    val_psnrs = []
    best_psnr = 0.0

    # Create directory for checkpoints
    os.makedirs('checkpoints', exist_ok=True)

    # Training loop
    #print("NUM_EPOCHS = ", NUM_EPOCHS )
    #print("model ==", model)
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss = train(model, train_loader, criterion, optimizer, epoch, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_psnr = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr)
        
        # Update learning rate
        scheduler.step()
        
        # Print statistics
        print(f'Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val PSNR: {val_psnr:.2f}dB')
        
        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_psnr': best_psnr,
            }, 'checkpoints/best_model.pth')
            print(f'New best model saved with PSNR: {best_psnr:.2f}dB')
        
        # Save model periodically
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f'checkpoints/model_epoch_{epoch+1}.pth')
    

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_psnrs, label='Val PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.title('Validation PSNR')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    # Test the best model
    print("Loading best model for testing...")
    checkpoint = torch.load('checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Run test
    avg_psnr, avg_ssim = test(model, test_loader, device, save_path='results')
    print(f"Test Results - Average PSNR: {avg_psnr:.2f}dB, Average SSIM: {avg_ssim:.4f}")

 # Run test
   # avg_psnr, avg_ssim = test(model, test_loader, device, save_path='results')
   # print(f"Test Results - Average PSNR: {avg_psnr:.2f}dB, Average SSIM: {avg_ssim:.4f}")

    # 🔹 Call visualization after testing
    visualize_sr_results(model, test_loader, device)


#%% Cell 6
if __name__ == "__main__":
    
    main()