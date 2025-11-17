import argparse
import numpy as np
import pandas as pd
import os
import random
import torch
import torch.nn as nn
import datetime
import time
import matplotlib.pyplot as plt
from torchinfo import summary
import yaml
import json
import sys
import glob
import copy
from tqdm import tqdm, trange

sys.path.append("..")
from lib.utils import (
    MaskedMAELoss,
    MaskedHuberLoss,
    print_log,
    seed_everything,
    set_cpu_num,
    masked_mae_loss,
    CustomJSONEncoder,
)
from lib.metrics import RMSE_MAE_MAPE
from lib.data_prepare import get_dataloaders_from_index_data, load_inrix_data
from model.MSGWTCN import GWTCN
from lib.utils1 import GetWavelet

# ! X shape: (B, T, N, C)


@torch.no_grad()
def eval_model(model, valset_loader, criterion):
    model.eval()
    batch_loss_list = []
    val_bar = tqdm(valset_loader, leave=False)
    for x_batch, y_batch in val_bar:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        B, T, N, C = y_batch.shape

        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)
        # Per-sample validity checks (flatten per sample to 2D: B x (N*T*1))
        pred_flat = out_batch.reshape(out_batch.shape[0], -1)
        true_flat = y_batch.reshape(y_batch.shape[0], -1)

        finite_pred = torch.isfinite(pred_flat).all(dim=1)   # (B,)
        finite_true = torch.isfinite(true_flat).all(dim=1)   # (B,)
        mag_ok_pred = pred_flat.abs().max(dim=1).values < 1e3
        mag_ok_true = true_flat.abs().max(dim=1).values < 1e3

        good_mask = finite_pred & finite_true & mag_ok_pred & mag_ok_true  # (B,)

        non_empty_mask = ~(y_batch == 0).all(dim=1).all(dim=2)  # (B,N)
        valid_mask = good_mask[:, None, None, None] * non_empty_mask[:, None, :, None]
        valid_mask = valid_mask.expand(B, T, N, C)

        bad_mask = ~good_mask
        loss = criterion(out_batch[valid_mask], y_batch[valid_mask])
        batch_loss_list.append(loss.item())
        val_bar.set_description(f"val loss:{loss.item():.5f} bad count:{bad_mask.sum().item()}")

    return np.mean(batch_loss_list)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    y = []
    out = []

    for x_batch, y_batch in tqdm(loader):
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)
    _, _, num_nodes, _ = out_batch.shape
    out = np.vstack(out).reshape(-1, 1, num_nodes)  # (samples, out_steps, num_nodes)
    y = np.vstack(y).reshape(-1, 1, num_nodes)

    pred_finite = np.isfinite(out).all(axis=(1, 2))   # (B,)
    true_finite = np.isfinite(y).all(axis=(1, 2))     # (B,)

    mag_ok_pred = np.abs(out).max(axis=(1, 2)) < 1e3  # (B,)
    mag_ok_true = np.abs(y).max(axis=(1, 2)) < 1e3    # (B,)

    good_mask = pred_finite & true_finite & mag_ok_pred & mag_ok_true  # (B,)
    good_y = y[good_mask]
    good_out = out[good_mask]
    return good_y, good_out


def train_one_epoch(
    model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=None
):
    global cfg, global_iter_count, global_target_length

    model.train()
    batch_loss_list = []
    train_bar = tqdm(trainset_loader, leave=False)
    for x_batch, y_batch in train_bar:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        B, T, N, C = y_batch.shape

        out_batch = model(x_batch)
        # print (f"-- output shape: {out_batch.shape} --")
        # print (f"-- label shape: {y_batch.shape} --")
        out_batch = SCALER.inverse_transform(out_batch)

        # Per-sample validity checks (flatten per sample to 2D: B x (N*T*1))
        pred_flat = out_batch.reshape(out_batch.shape[0], -1)
        true_flat = y_batch.reshape(y_batch.shape[0], -1)

        finite_pred = torch.isfinite(pred_flat).all(dim=1)   # (B,)
        finite_true = torch.isfinite(true_flat).all(dim=1)   # (B,)
        mag_ok_pred = pred_flat.abs().max(dim=1).values < 1e3
        mag_ok_true = true_flat.abs().max(dim=1).values < 1e3

        good_mask = finite_pred & finite_true & mag_ok_pred & mag_ok_true  # (B,)
        bad_mask = ~good_mask

        non_empty_mask = ~(y_batch == 0).all(dim=1).all(dim=2)  # (B,N)
        valid_mask = good_mask[:, None, None, None] * non_empty_mask[:, None, :, None]
        valid_mask = valid_mask.expand(B, T, N, C)

        loss = criterion(out_batch[valid_mask], y_batch[valid_mask])
        batch_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        train_bar.set_description(f"train loss:{loss.item():.5f} bad count:{bad_mask.sum().item()}/{B} empty count:{(~non_empty_mask).sum().item()}/{B*N}")

    epoch_loss = np.mean(batch_loss_list)
    scheduler.step()

    return epoch_loss


def train(
    model,
    trainset_loader,
    valset_loader,
    testset_loader,
    optimizer,
    scheduler,
    criterion,
    clip_grad=0,
    max_epochs=200,
    early_stop=10,
    verbose=1,
    plot=False,
    log=None,
    save=None,
):
    model = model.to(DEVICE)

    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(
            model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=log
        )
        train_loss_list.append(train_loss)

        val_loss = eval_model(model, valset_loader, masked_mae_loss)
        val_loss_list.append(val_loss)

        test_loss = eval_model(model, testset_loader, masked_mae_loss)

        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                " \tTrain Loss = %.5f" % train_loss,
                "Val Loss = %.5f" % val_loss,
                "Test Loss = %.5f" % test_loss,
                log=log,
            )

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= early_stop:
                break

    model.load_state_dict(best_state_dict)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, trainset_loader))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, valset_loader))

    out_str = f"Early stopping at epoch: {epoch+1}\n"
    out_str += f"Best at epoch {best_epoch+1}:\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        train_rmse,
        train_mae,
        train_mape,
    )
    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
    out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
        val_rmse,
        val_mae,
        val_mape,
    )
    print_log(out_str, log=log)

    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    if save:
        torch.save(best_state_dict, save)
    return model


@torch.no_grad()
def test_model(model, testset_loader, log=None):
    model.eval()
    print_log("--------- Test ---------", log=log)

    start = time.time()
    y_true, y_pred = predict(model, testset_loader)
    end = time.time()

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )
    # print (f"--- y_true: {y_true.shape}  y_pred: {y_pred.shape} ---")
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            i + 1,
            rmse,
            mae,
            mape,
        )

    print_log(out_str, log=log, end="")
    print_log("Inference time: %.2f s" % (end - start), log=log)


if __name__ == "__main__":
    # -------------------------- set running environment ------------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="GWTCN_INRIX_MANHATTAN")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    parser.add_argument("-m", "--mode", type=str, default="train")
    parser.add_argument("-s", "--shift", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--scale", type=float, default=0.85)
    parser.add_argument("--scale-offsets", type=float, nargs='+', default=[0,3,5])
    parser.add_argument("--approximation-order", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    args = parser.parse_args()

    # ------------------------------- configure random seeds ------------------------------ # 
    seed = random.randint(0,1000)  # set random seed here
    seed_everything(seed)
    set_cpu_num(1)

    GPU_ID = args.gpu_num
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    dataset = dataset.upper()
    data_path = f"../data/{dataset}"

    model_name = GWTCN.__name__

    with open(f"{model_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"../logs/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model_name}-{dataset}-{now}.log")
    log = open(log, "a", encoding="utf-8")
    log.seek(0)
    log.truncate()

    print_log(dataset, log=log)
    if 'INRIX' in args.dataset:
        (trainset_loader, valset_loader, testset_loader, SCALER, adj_mx) = (
            load_inrix_data(
                "/home/dachuan/Productivities/Spectral GAT/NY/adj_manhattan.npy",
                "/home/dachuan/Productivities/Spectral GAT/SPGAT/Data/speed_19_Manhattan_5min_py36",
                "/home/dachuan/Productivities/Spectral GAT/NY/Manhattan_FinalVersion.shp",
                "/home/dachuan/Productivities/Spectral GAT/NY/TMC_FinalVersion.csv",
                tod=cfg.get("time_of_day"),
                dow=cfg.get("day_of_week"),
                batch_size=cfg.get("batch_size", 64),
                history_seq_len=cfg.get("in_steps"),
                future_seq_len=cfg.get("out_steps"),
                log=log,
                train_ratio=cfg.get("train_size", 0.6),
                valid_ratio=cfg.get("val_size", 0.2),
                shift=args.shift,
            )
        )

    else:
        (trainset_loader, valset_loader, testset_loader, SCALER, adj_mx) = (
            get_dataloaders_from_index_data(
                data_path,
                tod=cfg.get("time_of_day"),
                dow=cfg.get("day_of_week"),
                batch_size=cfg.get("batch_size", 64),
                log=log,
                train_ratio=cfg.get("train_size", 0.6),
                valid_ratio=cfg.get("val_size", 0.2),
                shift=args.shift,
            )
        )

    print_log(log=log)
    supports = [torch.tensor(i).to(DEVICE) for i in adj_mx]

    # ---------------------- set up graph wavelets ---------------------- #
    scales = np.array(args.scale_offsets) + args.scale
    #Original GWNN version: fixed scales
    sparsifier = GetWavelet(adj_mx[0], args.scale, args.approximation_order, args.tolerance, scale_offsets=args.scale_offsets)
    #SPGAT: low and high waveletcoeff
    sparsifier.calculate_all_wavelets()
    num_nodes = adj_mx[0].shape[0]

    phi_matrices = torch.tensor(sparsifier.phi_matrices[0::2], dtype=torch.float32, device=DEVICE)
    inverse_phi_matrices = torch.tensor(sparsifier.phi_matrices[1::2], dtype=torch.float32, device=DEVICE)

    # print (f"--{[supports[i].shape for i in range(len(supports))]}--")

    # ---------------------- set loss, optimizer, scheduler ---------------------- #
    from functools import partial

    model = partial(GWTCN, num_nodes=num_nodes, wavelet_phi_matrices=phi_matrices, wavelet_inverse_phi_matrices=inverse_phi_matrices)
    model = model(**cfg["model_args"])
    criterion = MaskedMAELoss()  # MaskedHuberLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["milestones"],
        gamma=cfg.get("lr_decay_rate"),
        verbose=False,
    )
    # --------------------------- set model saving path -------------------------- #

    save_path = f"../saved_models/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.mode == "train":
        save = os.path.join(save_path, f"{model_name}-{dataset}-{now}.pt")
    elif args.mode == "test":
        model_files = glob.glob(os.path.join(save_path, f"{model_name}-{dataset}-*.pt"))
        if not model_files:
            raise ValueError("No saved model found for testing.")
        latest_model = max(model_files, key=os.path.getctime)
        print_log(f"Loading the latest model: {latest_model}", log=log)
        model.load_state_dict(torch.load(latest_model))
        model = model.to(DEVICE)

    # --------------------------- print model structure -------------------------- #

    print_log("---------", model_name, "---------", log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )
    print_log(
        summary(
            model,
            [
                cfg["batch_size"],
                cfg["in_steps"],
                cfg["num_nodes"],
                next(iter(trainset_loader))[0].shape[-1],
            ],
            verbose=0,  # avoid print twice
        ),
        log=log,
    )
    print_log(log=log)

    # --------------------------- train and test model --------------------------- #

    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)

    if args.mode == "train":
        model = train(
            model,
            trainset_loader,
            valset_loader,
            testset_loader,
            optimizer,
            scheduler,
            criterion,
            clip_grad=cfg.get("clip_grad"),
            max_epochs=cfg.get("max_epochs", 200),
            early_stop=cfg.get("early_stop", 10),
            verbose=1,
            log=log,
            save=save,
        )
        print_log(f"Saved Model: {save}", log=log)

    test_model(model, testset_loader, log=log)

    log.close()
