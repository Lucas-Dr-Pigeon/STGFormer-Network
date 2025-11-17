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
from lib.data_prepare import get_dataloaders_from_index_data, load_inrix_data_with_details
from model.STGWformer import STGWformer, noflayer

# ! X shape: (B, T, N, C)


@torch.no_grad()
def inference_graph(model):
    graph = torch.matmul(model.adaptive_embedding, model.adaptive_embedding.transpose(1, 2))
    graph = model.pooling(graph.transpose(0, 2)).transpose(0, 2)
    graph = nn.functional.relu(graph)
    graph = nn.functional.softmax(graph, dim=-1)
    return graph


def eval_model(model, valset_loader, criterion):
    model.eval()
    batch_loss_list = []
    for x_batch, y_batch in valset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

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

    return y, out


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
        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)

        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        train_bar.set_description(f"train loss:{loss.item():.5f}")

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


# ---------- 0) Utilities ----------

def _ensure_alignment(gdf, model):
    gdf = gdf.reset_index(drop=True)
    N = len(gdf)
    assert getattr(model, "num_nodes", N) == N, \
        f"Model num_nodes ({getattr(model, 'num_nodes', None)}) != len(gdf) ({N}). " \
        "If they differ, provide a mapping list from model index -> gdf row."
    return gdf, N

def _select_index(i, N, one_based=False):
    idx = i - 1 if one_based else i
    if not (0 <= idx < N):
        raise IndexError(f"Selected index {idx} out of range [0, {N-1}].")
    return idx

def _plot_spillover_positional(
    gdf,
    scores,
    source_idx,
    title='Spillover map',
    scale='z',                # 'raw' | 'log' | 'z'
    clip=(5, 95),             # percentile clip for color contrast
    cmap='plasma',
    highlight_color='cyan',
    topk_labels=10,           # annotate top-k influenced roads (set 0 to disable)
    show_hist=True,           # add a small histogram inset
    figsize=(10, 10)
):
    """
    gdf: GeoDataFrame (no node_id col needed; uses row order)
    scores: array-like, shape (N,)
    source_idx: int, positional index of source road (0-based)
    """

    df = gdf.reset_index(drop=True).copy()
    scores = np.asarray(scores, dtype=float)
    assert len(df) == len(scores), f"len(gdf)={len(df)} vs len(scores)={len(scores)}"

    # --- 1) Scaling ---
    x = scores.copy()
    if scale == 'log':
        # shift-to-positive then log1p
        shift = max(0.0, -np.min(x) + 1e-8)
        x = np.log1p(x + shift)
    elif scale == 'z':
        mu = np.nanmedian(x)
        sigma = np.nanstd(x)
        sigma = sigma if sigma > 1e-12 else 1.0
        x = (x - mu) / sigma

        # make strictly non-negative for color (optional)
        x = x - np.nanmin(x)
    elif scale == 'raw':
        pass
    else:
        raise ValueError("scale must be one of {'raw','log','z'}")

    df['spillover'] = x

    # --- 2) Percentile clipping for vmax ---
    lo, hi = clip
    vmin = np.nanpercentile(df['spillover'], lo)
    vmax = np.nanpercentile(df['spillover'], hi)
    if not np.isfinite(vmin): vmin = np.nanmin(df['spillover'])
    if not np.isfinite(vmax): vmax = np.nanmax(df['spillover'])
    if vmax <= vmin:  # fallback
        vmin, vmax = float(np.nanmin(df['spillover'])), float(np.nanmax(df['spillover']))
    vmin = max(vmin, 0.0)  # keep color scale [0, ...] for readability

    # --- 3) Plot base choropleth ---
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    df.plot(
        column='spillover',
        cmap=cmap,
        linewidth=2,
        legend=True,
        vmin=vmin,
        vmax=vmax,
        ax=ax
    )

    # --- 4) Highlight source link ---
    if 0 <= source_idx < len(df):
        df.iloc[[source_idx]].plot(
            ax=ax, color=highlight_color, linewidth=3, label=f'source idx={source_idx}'
        )
        ax.legend(frameon=True)
    else:
        print(f"[warn] source_idx {source_idx} out of range 0..{len(df)-1}")

    # --- 5) Label top-k influenced roads (by original scores, not scaled) ---
    if topk_labels and topk_labels > 0:
        order = np.argsort(scores)[::-1]
        top_idx = [i for i in order if i != source_idx][:topk_labels]
        # Use representative points to avoid label placement errors on LineStrings
        reps = df.geometry.representative_point()
        for i in top_idx:
            xy = reps.iloc[i].coords[0]
            ax.text(xy[0], xy[1], str(i), fontsize=7, ha='center', va='center')

    # --- 6) Optional histogram inset ---
    if show_hist:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax, width="35%", height="30%", loc='lower left', borderpad=1.3)
        axins.hist(scores[~np.isnan(scores)], bins=40)
        axins.set_title("Score dist.", fontsize=9)
        axins.tick_params(axis='both', labelsize=8)

    ax.set_title(f"{title}\n(scale={scale}, clip={clip} pctl)")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

# ---------- 1) Minimal taps into noflayer to read hop updates ----------
from contextlib import contextmanager

class _WaveletTap:
    def __init__(self): self.last = {}
    def clear(self): self.last.clear()

_WTAP = _WaveletTap()

def _tap_attention(self, x_BTNC, A_TNN):
    # Same math as STGWformer.noflayer.attention, but stash U
    B,T,N,C = x_BTNC.shape
    a1 = self.a[:self.in_features,:]
    a2 = self.a[self.in_features:,:]
    e = self.leakyrelu(torch.matmul(x_BTNC, a1) + torch.matmul(x_BTNC, a2).transpose(-2,-1))
    A_BTNN = A_TNN.unsqueeze(0).expand(B,-1,-1,-1)
    mask = (A_BTNN > 0)
    neg_inf = torch.finfo(e.dtype).min
    e_masked = torch.where(mask, e, e.new_full((), neg_inf))
    U = torch.softmax(e_masked, dim=-1)
    P = 0.5 * U
    _WTAP.last['U'] = U.detach()
    _WTAP.last['A'] = A_BTNN.detach()
    return U, P, A_BTNN

def _tap_forward_lifting_bases(self, x_BTNC, P_BTNN, U_BTNN, A_BTNN):
    # Same as STGWformer.noflayer.forward_lifting_bases, stash per-hop updates
    B,T,N,C = x_BTNC.shape
    coe = torch.sigmoid(self.temp)
    cheb_coe = torch.sigmoid(self.cheb)
    AdjP = A_BTNN * P_BTNN
    rowsum = AdjP.sum(-1)
    update = x_BTNC
    feat_prime = None
    per_hop_updates = []

    for step in range(self.hop):
        update = torch.einsum("btij,btjc->btic", U_BTNN, update)
        per_hop_updates.append(update.detach())

        if self.alpha_ is None:
            feat_even_bar = coe[0]*x_BTNC + update
        else:
            feat_even_bar = update

        if step >= 1:
            rowsum = cheb_coe[step-1] * rowsum

        feat_odd_bar = update - feat_even_bar * rowsum.unsqueeze(-1)

        if step == 0:
            if self.alpha_ is None:
                feat_fuse = coe[1]*feat_even_bar + (1-coe[1])*feat_odd_bar
                feat_prime = coe[2]*x_BTNC + (1-coe[2])*feat_fuse
            else:
                feat_fuse = self.alpha_*feat_even_bar + (1-self.alpha_)*feat_odd_bar
                feat_prime = self.alpha_*x_BTNC + (1-self.alpha_)*feat_fuse
        else:
            if self.alpha_ is None:
                feat_fuse = coe[1]*feat_even_bar + (1-coe[1])*feat_odd_bar
                feat_prime = coe[2]*feat_prime + (1-coe[2])*feat_fuse
            else:
                feat_fuse = self.alpha_*feat_even_bar + (1-self.alpha_)*feat_odd_bar
                feat_prime = self.alpha_*feat_prime + (1-self.alpha_)*feat_fuse

    _WTAP.last['per_hop_updates'] = per_hop_updates
    return feat_prime

@contextmanager
def _patch_noflayer_for_tap(noflayer_cls):
    orig_attn = noflayer_cls.attention
    orig_lift = noflayer_cls.forward_lifting_bases
    noflayer_cls.attention = _tap_attention
    noflayer_cls.forward_lifting_bases = _tap_forward_lifting_bases
    try:
        yield
    finally:
        noflayer_cls.attention = orig_attn
        noflayer_cls.forward_lifting_bases = orig_lift

# ---------- 2) Spillover scores using positional index ----------
@torch.no_grad()
def hop_spillover_scores_positional(model, batch_x, idx_source,
                                    channel_idx=0, t_in=None, eps=-5.0):
    """
    idx_source: integer positional index in gdf/data order.
    Returns list-of-arrays: scores_per_hop[k] shape (N,)
    """
    model.eval()
    device = next(model.parameters()).device
    x0 = batch_x.to(device)

    if t_in is None:
        t_in = x0.shape[1]-1  # last observed input step

    # Baseline
    _WTAP.clear()
    with _patch_noflayer_for_tap(noflayer):
        _ = model(x0)
        per_hop_base = _WTAP.last['per_hop_updates']  # list of (B,T,N,C)

    # Perturb the source road at time t_in
    x1 = x0.clone()
    x1[:, t_in, idx_source, channel_idx] += eps
    _WTAP.clear()
    with _patch_noflayer_for_tap(noflayer):
        _ = model(x1)
        per_hop_pert = _WTAP.last['per_hop_updates']

    scores_per_hop = []
    denom = abs(eps) + 1e-8
    for base, pert in zip(per_hop_base, per_hop_pert):
        delta = (pert - base) / denom         # (B,T,N,C)
        node_score = torch.linalg.vector_norm(delta, dim=-1).mean(dim=(0,1))  # (N,)
        scores_per_hop.append(node_score.detach().cpu().numpy())
    print (f'--{scores_per_hop[0].mean(), scores_per_hop[0].min(), scores_per_hop[0].max()}--')
    return scores_per_hop

# ---------- 3) Forecast sensitivity (end-to-end) ----------
@torch.no_grad()
def forecast_delta_map_positional(model, batch_x, idx_source,
                                  channel_idx=0, t_in=None, eps=-5.0, horizon_h=0):
    device = next(model.parameters()).device
    x0 = batch_x.to(device)
    if t_in is None: t_in = x0.shape[1]-1
    y0 = model(x0)  # (B, H, N, 1)

    x1 = x0.clone()
    x1[:, t_in, idx_source, channel_idx] += eps
    y1 = model(x1)

    dy = (y1 - y0)[:, horizon_h, :, 0].mean(dim=0) / (abs(eps) + 1e-8)  # (N,)
    return dy.abs().cpu().numpy()


def _gamma_enhance(x, gamma=0.4):
    x_min, x_max = np.nanmin(x), np.nanmax(x)
    x_scaled = (x - x_min) / (x_max - x_min + 1e-9)
    return np.sign(x_scaled) * (np.abs(x_scaled) ** gamma)

# ---------- 4) End-to-end example pipeline ----------
def visualize_spillover_from_index(gdf, trainset_loader, model,
                                   index_number=52, one_based=False,
                                   channel_idx=0, eps=-5.0, horizon_h=0):
    gdf, N = _ensure_alignment(gdf, model)
    idx = _select_index(index_number, N, one_based=one_based)

    batch_x, _ = next(iter(trainset_loader))  # expect [B, T, N, F]
    # Optional: verify shapes
    assert batch_x.shape[2] == N, f"Batch N={batch_x.shape[2]} != len(gdf)={N}"

    # Wavelet hop spillover
    scores_per_hop = hop_spillover_scores_positional(
        model, batch_x, idx_source=idx, channel_idx=channel_idx, eps=eps
    )

    # Plot hop-1 and hop-2 (if available)
    _plot_spillover_positional(
        gdf, _gamma_enhance(scores_per_hop[0]), idx,
        title=f'SEAGWNN hop-1 spillover (source idx={index_number}, one_based={one_based})'
    )


    if len(scores_per_hop) > 1:
        _plot_spillover_positional(
            gdf, _gamma_enhance(scores_per_hop[1]), idx,
            title=f'SEAGWNN hop-2 spillover (source idx={index_number}, one_based={one_based})'
        )

    # End-to-end forecast sensitivity at horizon 0
    sens = forecast_delta_map_positional(
        model, batch_x, idx_source=idx, channel_idx=channel_idx, eps=eps, horizon_h=horizon_h
    )
    _plot_spillover_positional(
        gdf, sens, idx,
        title=f'Forecast sensitivity (h={horizon_h}) (source idx={index_number}, one_based={one_based})'
    )


if __name__ == "__main__":
    # -------------------------- set running environment ------------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="STGWformer_INRIX_MANHATTAN")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    parser.add_argument("-m", "--mode", type=str, default="test")
    parser.add_argument("-s", "--shift", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:1")
    args = parser.parse_args()

    seed = random.randint(0,1000)  # set random seed here
    seed_everything(seed)
    set_cpu_num(1)

    GPU_ID = args.gpu_num
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = args.dataset
    dataset = dataset.upper()
    data_path = f"../data/{dataset}"

    model_name = STGWformer.__name__

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

    # ------------------------------- load dataset ------------------------------- #

    print_log(dataset, log=log)
    if 'INRIX' in args.dataset:
        (trainset_loader, valset_loader, testset_loader, SCALER, adj_mx, gdf, tmc) = (
            load_inrix_data_with_details(
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

    # print (f"--{[supports[i].shape for i in range(len(supports))]}--")

    # ---------------------- set loss, optimizer, scheduler ---------------------- #
    from functools import partial

    model = partial(STGWformer)
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
            device=DEVICE,
        ),
        log=log,
    )
    print_log(log=log)


    # --------------------------- inference graph --------------------------- #

    graph = inference_graph(model)

    # scores_per_hop, U = hop_spillover_scores_positional(model, batch_x, node_i=190, channel_idx=0, eps=-5.0)


    visualize_spillover_from_index(gdf, trainset_loader, model,
                                   index_number=0, one_based=False,
                                   channel_idx=0, eps=-0.2, horizon_h=0)
    

    # --------------------------- train and test model --------------------------- #

    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)
    # test_model(model, testset_loader, log=log)




    log.close()
