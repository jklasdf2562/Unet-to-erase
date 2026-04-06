import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from U.config import (
    device,
    DATA_DIRS,
    INPUT_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LR,
    WEIGHT_DECAY,
    CHECKPOINT_PATH,
    BEST_MODEL_PATH,
    SEED,
)
from U.data.dataset import collect_data_pair, ImageDataset, compute_mean_std
from U.models.unet import Unet
from U.utils import denormalize, tensor_to_uint8_image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def train_model(train_loader, val_loader, model, means, stds):
    criterion_l1 = nn.L1Loss()
    critertion_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    schedeler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_loss = float('inf')
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        schedeler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', best_loss)
        print(f"Resuming training from epoch {start_epoch}, best_loss={best_loss:.4f}")
    patience_counter = 0
    patience = 10
    train_losses = []
    val_losses = []
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_l1 = criterion_l1(outputs, targets)
            loss = loss_l1 + 0.5 * (1 - critertion_ssim(outputs, targets))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        psnr.reset()
        ssim.reset()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion_l1(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                psnr.update(outputs, targets)
                ssim.update(outputs, targets)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        avg_psnr = psnr.compute().item()
        avg_ssim = ssim.compute().item()
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}, Val PSNR: {avg_psnr:.4f}, Val SSIM: {avg_ssim:.4f}")
        schedeler.step(val_loss)

        # save sample images
        if (epoch + 1) % 1 == 0:
            model.eval()
            cnt = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    if inputs.size(0) > 0:
                        indices = random.sample(range(inputs.size(0)), min(2, inputs.size(0)))
                        for i in indices:
                            inp = denormalize(inputs[i], means, stds).squeeze(0)
                            out = outputs[i].squeeze(0)
                            tgt = targets[i].squeeze(0)
                            inp_img = tensor_to_uint8_image(inp)
                            out_img = tensor_to_uint8_image(out)
                            tgt_img = tensor_to_uint8_image(tgt)
                            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                            axs[0].imshow(inp_img)
                            axs[0].set_title('Input')
                            axs[1].imshow(out_img)
                            axs[1].set_title('Output')
                            axs[2].imshow(tgt_img)
                            axs[2].set_title('Target')
                            for ax in axs:
                                ax.axis('off')
                            plt.tight_layout()
                            plt.savefig(f'epoch{epoch+1}_sample{cnt+1}.png')
                            plt.close()
                            cnt += 1
                            if cnt >= 2:
                                break
                    if cnt >= 2:
                        break

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': schedeler.state_dict(),
            'best_loss': best_loss,
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print("Model saved!")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    # 绘制loss曲线
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    data_pairs = collect_data_pair(DATA_DIRS)
    print(f"Total data pairs collected: {len(data_pairs)}")
    train_pairs, val_pairs = train_test_split(data_pairs, test_size=0.2, random_state=SEED)

    means, stds = compute_mean_std(train_pairs)

    train_dataset = ImageDataset(train_pairs, input_size=INPUT_SIZE, mean=means, std=stds, is_train=True)
    val_dataset = ImageDataset(val_pairs, input_size=INPUT_SIZE, mean=means, std=stds, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = Unet(image_size=INPUT_SIZE).to(device)
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        print(f"Loaded pretrained model from {BEST_MODEL_PATH}")
    else:
        print("No pretrained model found, training from scratch.")

    train_model(train_loader, val_loader, model, means, stds)


if __name__ == '__main__':
    main()
