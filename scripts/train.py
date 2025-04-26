import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingWarmRestarts

from torchvision import transforms, datasets
from torchvision.datasets import CIFAR10, ImageFolder
import torchvision.transforms.v2 as v2
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from evaluate import evaluate
import math

import re


from functools import partial
from typing import List

def train_rdnet_tiny(model, train_loader, val_loader, test_loader, device, num_epochs=20, 
                    initial_lr=1e-3, weight_decay=0.05, betas=(0.9, 0.999), 
                    save_interval=10, resume_path=None, model_name_prefix='rdnet_tiny_transfer_learn',
                    early_stopping_patience=10, run_name="model", sub_dir_seed_number='seed_29'):
    
    model.to(device)
    warmup_epochs = int(math.ceil((10/100)*num_epochs))
    # Optimizer and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, betas=betas, weight_decay=weight_decay)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=1, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    # Loss function
    criterion = nn.CrossEntropyLoss()


    best_val_loss = float('inf')
    correspondin_val_acc = 0.0
    start_epoch = 0
    early_stopping_counter = 0

    history = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    # Resume from checkpoint
    if resume_path and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        history = checkpoint['history']
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = history['val_loss'][-1]
        correspondin_val_acc = history['val_acc'][-1]
        print(f"Resuming training from epoch {start_epoch}")

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Per-iteration print
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Batch [{batch_idx+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, "
                    f"Running Acc: {100. * correct / total:.2f}%")

            train_loss = running_loss / total
            train_acc = 100. * correct / total


            
            val_loss, val_acc = evaluate(model, val_loader, device)


            test_loss, test_acc = evaluate(model, test_loader, device)

            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Test Loss: {test_loss:.4f} | "
                  f"Test Acc: {test_acc:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                correspondin_val_acc = val_acc
                early_stopping_counter = 0
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'history': history
                }
                os.makedirs(f'{run_name}/{sub_dir_seed_number}', exist_ok=True)
                save_path = os.path.join(f'{run_name}/{sub_dir_seed_number}', f"{model_name_prefix}__valLoss{val_loss:.4f}_valAcc{correspondin_val_acc:.2f}.pth")
                torch.save(checkpoint, save_path)
                print(f"Saved best checkpoint at {save_path}")
            else:
                early_stopping_counter += 1
                print(f"No improvement in validation loss for {early_stopping_counter} epoch(s).")

            scheduler.step()
            
            # Save periodic checkpoints
            if epoch % save_interval == 0:
                os.makedirs(f'{run_name}/{sub_dir_seed_number}/periodic', exist_ok=True)
                save_path = os.path.join(f'{run_name}/{sub_dir_seed_number}/periodic', f"{model_name_prefix}_epoch{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'history': history
                }, save_path)
                print(f"Saved periodic checkpoint at {save_path}")

            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                os.makedirs(f'{run_name}/{sub_dir_seed_number}/earlystopping', exist_ok=True)
                save_path = os.path.join(f'{run_name}/{sub_dir_seed_number}/earlystopping', f"{model_name_prefix}_epoch{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'history': history
                }, save_path)
                print(f"Saved periodic checkpoint at {save_path}")
                break

        os.makedirs(f'{run_name}/{sub_dir_seed_number}/last_epoch', exist_ok=True)
        save_path = os.path.join(f'{run_name}/{sub_dir_seed_number}/last_epoch', f"{model_name_prefix}_epoch{epoch}.pth")
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history
        }, save_path)
        return history, correspondin_val_acc

    except KeyboardInterrupt:
        print("Training interrupted! Saving checkpoint...")
        os.makedirs(f'{run_name}/{sub_dir_seed_number}/KeyboardInterrupt', exist_ok=True)
        save_path = os.path.join(f'{run_name}/{sub_dir_seed_number}/KeyboardInterrupt', f"{model_name_prefix}_interrupted_epoch{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history
        }, save_path)
        print(f"Checkpoint saved at {save_path}. You can resume training from here.")
        return history, correspondin_val_acc
    
