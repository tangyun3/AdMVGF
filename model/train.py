import argparse
import numpy as np
import os
import torch
import copy
import yaml
import sys
from tqdm import tqdm

sys.path.append("..")
from lib.utils import MaskedHuberLoss, seed_everything, masked_mae_loss
from lib.metrics import RMSE_MAE_MAPE
from lib.data_prepare import get_dataloaders_from_index_data
from model.AdMVGF import AdMVGF

# ! X shape: (B, T, N, C)

@torch.no_grad()
def eval_model(model, valset_loader, criterion, device, scaler):
    model.eval()
    batch_loss_list = []
    for x_batch, y_batch in valset_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        out_batch = model(x_batch)
        out_batch = scaler.inverse_transform(out_batch)
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())
    return np.mean(batch_loss_list)


@torch.no_grad()
def predict(model, loader, device, scaler):
    model.eval()
    y = []
    out = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        out_batch = model(x_batch)
        out_batch = scaler.inverse_transform(out_batch)
        out.append(out_batch.cpu().numpy())
        y.append(y_batch.cpu().numpy())
    out = np.vstack(out).squeeze()
    y = np.vstack(y).squeeze()
    return y, out


def train_one_epoch(model, trainset_loader, optimizer, criterion, clip_grad, device, scaler):
    model.train()
    batch_loss_list = []
    pbar = tqdm(trainset_loader, desc="Training", leave=False)

    for x_batch, y_batch in pbar:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        out_batch = model(x_batch)
        out_batch = scaler.inverse_transform(out_batch)

        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    return np.mean(batch_loss_list)


def train(
    model,
    trainset_loader,
    valset_loader,
    testset_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    scaler,
    clip_grad=0,
    max_epochs=200,
    early_stop=5
):
    model = model.to(device)
    wait = 0
    min_val_loss = np.inf
    best_state_dict = None

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        train_loss = train_one_epoch(model, trainset_loader, optimizer, criterion, clip_grad, device, scaler)
        val_loss = eval_model(model, valset_loader, masked_mae_loss, device, scaler)
        test_loss = eval_model(model, testset_loader, masked_mae_loss, device, scaler)
        scheduler.step()

        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Test Loss: {test_loss:.5f}")

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= early_stop:
                print(f"\nEarly stopping at epoch: {epoch+1}")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model


@torch.no_grad()
def test_model(model, testset_loader, device, scaler):
    y_true, y_pred = predict(model, testset_loader, device, scaler)
    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    print("\n========== Test Results ==========")
    print(f"RMSE  : {rmse_all:.5f}")
    print(f"MAE   : {mae_all:.5f}")
    print(f"MAPE  : {mape_all:.5f}")
    print("=================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="pems08")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    args = parser.parse_args()

    seed_everything(424)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset.upper()
    model_name = AdMVGF.__name__

    with open(f"{model_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

    data_path = f"../data/{dataset}"
    trainset_loader, valset_loader, testset_loader, SCALER, adj_mx = get_dataloaders_from_index_data(
        data_path,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        batch_size=cfg.get("batch_size", 64),
        train_ratio=cfg.get("train_size", 0.6),
        valid_ratio=cfg.get("val_size", 0.2),
    )
    supports = [torch.tensor(i).to(DEVICE) for i in adj_mx]
    model = AdMVGF(**cfg["model_args"], supports=supports)
    criterion = MaskedHuberLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["milestones"],
        gamma=cfg.get("lr_decay_rate", 0.1),
        verbose=False,
    )
    model = train(
        model,
        trainset_loader,
        valset_loader,
        testset_loader,
        optimizer,
        scheduler,
        criterion,
        DEVICE,
        SCALER,
        clip_grad=cfg.get("clip_grad"),
        max_epochs=cfg.get("max_epochs", 200),
        early_stop=cfg.get("early_stop", 5),
    )

    test_model(model, testset_loader, DEVICE, SCALER)