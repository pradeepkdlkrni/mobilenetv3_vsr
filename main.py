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
import argparse
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
from os import chdir, getcwd
import time

#%% Cell 2
# Import model components
from model import VideoSuperResolution
from dataset import VideoDataset
from loss import CombinedLoss
from train import train, validate, test, measure_inference_time
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
#print(f"Using device: {device}")
#%% Cell 5
# Define hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10

# Scale factor is now configurable via command-line argument
parser = argparse.ArgumentParser(description='Video Super Resolution with Dynamic Scaling')
parser.add_argument('--scale', type=int, default=4, choices=[2, 4, 8, 16, 32, 64],
                   help='Super-resolution scale factor (2, 4, 8, 16, 32, or 64)')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--sequence_length', type=int, default=5, help='Number of consecutive frames to process')

args = parser.parse_args()

# Update hyperparameters from command line
SCALE_FACTOR = args.scale
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
SEQUENCE_LENGTH = args.sequence_length

#print(f"Training with scale factor: {SCALE_FACTOR}x")
#print(f"Sequence length: {SEQUENCE_LENGTH}")
#print(f"Number of epochs: {NUM_EPOCHS}")
#print(f"Batch size: {BATCH_SIZE}")
#print(f"Learning rate: {LEARNING_RATE}")

#%%
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Define path to save SR images
os.makedirs('results', exist_ok=True)
'''
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
                lr_image = np.clip(lr_image * 255.0, 0, 255).astype(np.uint8)
                sr_image = np.clip(sr_image * 255.0, 0, 255).astype(np.uint8)
                hr_image = np.clip(hr_image * 255.0, 0, 255).astype(np.uint8)

                # Save images
                cv2.imwrite(f"results/lr_video{vid_idx}_frame{frame_idx}.png", cv2.cvtColor(lr_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"results/sr_video{vid_idx}_frame{frame_idx}_scale{SCALE_FACTOR}x.png", cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"results/hr_video{vid_idx}_frame{frame_idx}.png", cv2.cvtColor(hr_image, cv2.COLOR_RGB2BGR))

                # Display first few frames
                if vid_idx < num_videos and frame_idx < 3:
                    plt.figure(figsize=(15, 5))
                    plt.subplot(1, 3, 1)
                    plt.imshow(lr_image)
                    plt.title(f"Low-Res ({lr_image.shape[1]}x{lr_image.shape[0]})")
                    plt.axis("off")

                    plt.subplot(1, 3, 2)
                    plt.imshow(sr_image)
                    plt.title(f"Super-Res {SCALE_FACTOR}x ({sr_image.shape[1]}x{sr_image.shape[0]})")
                    plt.axis("off")

                    plt.subplot(1, 3, 3)
                    plt.imshow(hr_image)
                    plt.title(f"High-Res GT ({hr_image.shape[1]}x{hr_image.shape[0]})")
                    plt.axis("off")

                    plt.savefig(f"results/comparison_video{vid_idx}_frame{frame_idx}_scale{SCALE_FACTOR}x.png")
                    plt.show()
            
            if vid_idx == num_videos - 1:  # Stop after visualizing a few videos
                break
'''
def visualize_sr_results(model, test_loader, device, num_videos=3):
    model.eval()
    with torch.no_grad():
        for vid_idx, (lr_frames, hr_frames, lr_paths_batch) in enumerate(test_loader):  # lr_paths added!
            print("lr_paths_batch type:", type(lr_paths_batch))
            print("len(lr_paths_batch):", len(lr_paths_batch))
            print("lr_paths_batch[0]:", lr_paths_batch[0])

            lr_paths = lr_paths_batch  # Unpack single batch element
            lr_frames = lr_frames.to(device)
            sr_frames = model(lr_frames)

            for frame_idx in range(lr_frames.shape[1]):
                # Load original LR image from path
                lr_path = lr_paths[frame_idx][0]  # batch index 0
                original_lr_image = cv2.imread(lr_path)
                original_lr_image = cv2.cvtColor(original_lr_image, cv2.COLOR_BGR2RGB)

                sr_image = sr_frames[0, frame_idx].cpu().numpy().transpose(1, 2, 0)
                hr_image = hr_frames[0, frame_idx].cpu().numpy().transpose(1, 2, 0)

                sr_image = np.clip(sr_image * 255.0, 0, 255).astype(np.uint8)
                hr_image = np.clip(hr_image * 255.0, 0, 255).astype(np.uint8)

                # Save visuals
                cv2.imwrite(f"results/lr_orig_video{vid_idx}_frame{frame_idx}.png", cv2.cvtColor(original_lr_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"results/sr_video{vid_idx}_frame{frame_idx}_scale{SCALE_FACTOR}x.png", cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f"results/hr_video{vid_idx}_frame{frame_idx}.png", cv2.cvtColor(hr_image, cv2.COLOR_RGB2BGR))

                if vid_idx < num_videos and frame_idx < 3:
                    plt.figure(figsize=(15, 5))
                    plt.subplot(1, 3, 1)
                    plt.imshow(original_lr_image)
                    plt.title(f"Original LR ({original_lr_image.shape[1]}x{original_lr_image.shape[0]})")
                    plt.axis("off")

                    plt.subplot(1, 3, 2)
                    plt.imshow(sr_image)
                    plt.title(f"Super-Res {SCALE_FACTOR}x ({sr_image.shape[1]}x{sr_image.shape[0]})")
                    plt.axis("off")

                    plt.subplot(1, 3, 3)
                    plt.imshow(hr_image)
                    plt.title(f"High-Res GT ({hr_image.shape[1]}x{hr_image.shape[0]})")
                    plt.axis("off")

                    plt.savefig(f"results/comparison_video{vid_idx}_frame{frame_idx}_scale{SCALE_FACTOR}x.png")
                    plt.show()


#%% Cell 7
def main():
    # Create model
    print(f"Creating VSR model with {SCALE_FACTOR}x upscaling...")
    model = VideoSuperResolution(scale_factor=SCALE_FACTOR).to(device)
    
    # Define loss function
    criterion = CombinedLoss().to(device)

    # Define optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Define dataset paths
    base_path = "F://vimeo_septuplet_full//Arranged_small"
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
        transform=(train_transform_lr, train_transform_hr),
        scale_factor=SCALE_FACTOR
    )
    
    val_dataset = VideoDataset(
        lr_folder=val_lr_path,
        hr_folder=val_hr_path,
        sequence_length=SEQUENCE_LENGTH,
        transform=(valid_transform_lr, valid_transform_hr),
        scale_factor=SCALE_FACTOR
    )
    
    test_dataset = VideoDataset(
        lr_folder=test_lr_path,
        hr_folder=test_hr_path,
        sequence_length=SEQUENCE_LENGTH,
        transform=(valid_transform_lr, valid_transform_hr),
        scale_factor=SCALE_FACTOR
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Check sample batch shapes
    for lr_frames, hr_frames, lr_paths  in train_loader:
        print(f"LR input shape: {lr_frames.shape}")  # [batch_size, sequence_length, channels, height, width]
        print(f"HR target shape: {hr_frames.shape}")
        print(f"Scale factor verification: HR height / LR height = {hr_frames.shape[3] / lr_frames.shape[3]}")
        print(f"Scale factor verification: HR width / LR width = {hr_frames.shape[4] / lr_frames.shape[4]}")
        break  # Just print the first batch
    
    # Training log
    train_losses = []
    val_losses = []
    val_psnrs = []
    best_psnr = 0.0

    epoch_times = []  # Track epoch times

    # Create directory for checkpoints with scale factor in name
    checkpoint_dir = f'checkpoints_scale{SCALE_FACTOR}x'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    #print("NUM_EPOCHS = ", NUM_EPOCHS )
    #print("model ==", model)
    total_training_start  = time.time()
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss = train(model, train_loader, criterion, optimizer, epoch, device, SCALE_FACTOR)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_psnr = validate(model, val_loader, criterion, device, SCALE_FACTOR)
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
                'scale_factor': SCALE_FACTOR,
            }, f'{checkpoint_dir}/best_model.pth')
            print(f'New best model saved with PSNR: {best_psnr:.2f}dB')
        
        # Save model periodically
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scale_factor': SCALE_FACTOR,
            }, f'{checkpoint_dir}/model_epoch_{epoch+1}.pth')
    
    total_training_time = time.time() - total_training_start
  #  avg_epoch_time = sum(epoch_times) / len(epoch_times)
    
    print(f"\nTraining Performance Summary:")
    print(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
  #  print(f"Average time per epoch: {avg_epoch_time:.2f} seconds ({avg_epoch_time/60:.2f} minutes)")
   # print(f"Fastest epoch: {min(epoch_times):.2f} seconds")
   # print(f"Slowest epoch: {max(epoch_times):.2f} seconds")


    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training and Validation Loss ({SCALE_FACTOR}x)')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_psnrs, label='Val PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.title(f'Validation PSNR ({SCALE_FACTOR}x)')
    
    plt.tight_layout()
    plt.savefig(f'training_curves_scale{SCALE_FACTOR}x.png')
    plt.show()
    
    # Test the best model
    print("Loading best model for testing...")
    checkpoint = torch.load(f'{checkpoint_dir}/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create results directory with scale factor
    results_dir = f'results_scale{SCALE_FACTOR}x'
    os.makedirs(results_dir, exist_ok=True)
    
    # Run test
    avg_psnr, avg_ssim = test(model, test_loader, device,SCALE_FACTOR, save_path=results_dir)
    print(f"Test Results ({SCALE_FACTOR}x) - Average PSNR: {avg_psnr:.2f}dB, Average SSIM: {avg_ssim:.4f}")

    # After testing is complete, measure inference time
    print("\nMeasuring inference performance...")
    inference_time, avg_frame_time, fps = measure_inference_time(model, test_loader, device, SCALE_FACTOR)
    
    # Call visualization after testing
    visualize_sr_results(model, test_loader, device)


#%% Cell 6
if __name__ == "__main__":
    
    main()