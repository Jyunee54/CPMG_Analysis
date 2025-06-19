import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from imports.models import HPC, HPC_hierarchical, Regression_model
from sklearn.metrics import accuracy_score


def train_hpc_model(
        X_train_path,
        Y_train_path,
        X_valid_path,
        Y_valid_path,
        model_save_path,
        model_type='HPC',
        epochs=50,
        batch_size=128,
        learning_rate=0.001,
):
    print("\n[INFO] Loading data...")
    X_train = np.load(X_train_path)
    Y_train = np.load(Y_train_path)
    X_valid = np.load(X_valid_path)
    Y_valid = np.load(Y_valid_path)

    # Convert to PyTorch tensors
    X_train = torch.Tensor(X_train)
    Y_train = torch.Tensor(Y_train)
    X_valid = torch.Tensor(X_valid)
    Y_valid = torch.Tensor(Y_valid)

    if len(Y_train.shape) == 1 or Y_train.shape[1] == 1:
        Y_train = Y_train.long()
        Y_valid = Y_valid.long()
        loss_fn = nn.CrossEntropyLoss()
        num_classes = len(torch.unique(Y_train))
    else:
        loss_fn = nn.MSELoss()
        num_classes = Y_train.shape[1]

    input_dim = X_train.shape[1]

    print(f"[INFO] Model: {model_type} | Input dim: {input_dim} | Output dim: {num_classes}")

    # Model selection
    if model_type == 'HPC':
        model = HPC(input_dim, num_classes)
    elif model_type == 'HPC_hierarchical':
        model = HPC_hierarchical(input_dim, num_classes)
    elif model_type == 'Regression_model':
        model = Regression_model(input_dim, num_classes)
    else:
        raise ValueError("Invalid model_type. Choose from ['HPC', 'HPC_hierarchical', 'Regression_model']")

    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_valid, Y_valid), batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.cuda(), yb.cuda()

            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb) if isinstance(loss_fn, nn.MSELoss) else loss_fn(preds, yb.squeeze())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.cuda(), yb.cuda()
                preds = model(xb)
                loss = loss_fn(preds, yb) if isinstance(loss_fn, nn.MSELoss) else loss_fn(preds, yb.squeeze())
                val_loss += loss.item() * xb.size(0)

                if isinstance(loss_fn, nn.CrossEntropyLoss):
                    all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
                    all_labels.extend(yb.cpu().numpy())

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", end='')

        if isinstance(loss_fn, nn.CrossEntropyLoss):
            acc = accuracy_score(all_labels, all_preds) * 100
            print(f" | Val Acc: {acc:.2f}%")
        else:
            print()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print("[INFO] Best model saved.")


if __name__ == '__main__':
    # Example usage
    train_hpc_model(
        X_train_path='data/denoised_px/denoised_train.npy',
        Y_train_path='data/Y_train_class.npy',
        X_valid_path='data/denoised_px/denoised_valid.npy',
        Y_valid_path='data/Y_valid_class.npy',
        model_save_path='data/models/hpc_model_504.pt',
        model_type='HPC',
        epochs=30,
        batch_size=128,
        learning_rate=0.001
    )
