import numpy as np
import torch
import pickle
import random
import os
import json
from shapely.geometry import LineString, MultiLineString
from scipy.spatial import cKDTree  
import geopandas as gpd
from typing import Sequence, Tuple


class StandardScaler:
    """
    Standard the input
    https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py
    """

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit_transform(self, data):
        self.mean = data.mean()
        self.std = data.std()

        return (data - self.mean) / self.std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean
    
class MinMaxScaler:
    """
    Min-Max scaler to scale input to [0, 1] (or custom range).
    Based on structure from Graph-WaveNet's StandardScaler.
    """

    def __init__(self, min_val, max_val, min_feature=0, max_feature=1):
        self.min_val = min_val
        self.max_val = max_val
        self.min_feature = min_feature
        self.max_feature = max_feature

        self.val_range = max_val - min_val
        self.feature_range = max_feature - min_feature

    def transform(self, data):
        return (data - self.min_val) / self.val_range * self.feature_range + self.min_feature

    def inverse_transform(self, data):
        return (data - self.min_feature) / self.feature_range * self.val_range + self.min_val


def masked_mae_loss(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class MaskedMAELoss:
    def _get_name(self):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=0.0):
        return masked_mae_loss(preds, labels, null_val)


def masked_huber_loss(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.nn.functional.huber_loss(preds, labels, delta=1, reduction="none")
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


class MaskedHuberLoss:
    def _get_name(self, delta=0.5):
        return self.__class__.__name__

    def __call__(self, preds, labels, null_val=0.0):
        return masked_huber_loss(preds, labels, null_val)


def print_log(*values, log=None, end="\n"):
    print(*values, end=end)
    if log:
        if isinstance(log, str):
            log = open(log, "a")
        print(*values, file=log, end=end)
        log.flush()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def set_cpu_num(cpu_num: int):
    os.environ["OMP_NUM_THREADS"] = str(cpu_num)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
    os.environ["MKL_NUM_THREADS"] = str(cpu_num)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
    torch.set_num_threads(cpu_num)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return f"Shape: {obj.shape}"
        elif isinstance(obj, torch.device):
            return str(obj)
        else:
            return super(CustomJSONEncoder, self).default(obj)


def vrange(starts, stops):
    """Create ranges of integers for multiple start/stop

    Parameters:
        starts (1-D array_like): starts for each range
        stops (1-D array_like): stops for each range (same shape as starts)

        Lengths of each range should be equal.

    Returns:
        numpy.ndarray: 2d array for each range

    For example:

        >>> starts = [1, 2, 3, 4]
        >>> stops  = [4, 5, 6, 7]
        >>> vrange(starts, stops)
        array([[1, 2, 3],
               [2, 3, 4],
               [3, 4, 5],
               [4, 5, 6]])

    Ref: https://codereview.stackexchange.com/questions/83018/vectorized-numpy-version-of-arange-with-multiple-start-stop
    """
    stops = np.asarray(stops)
    l = stops - starts  # Lengths of each range. Should be equal, e.g. [12, 12, 12, ...]
    assert l.min() == l.max(), "Lengths of each range should be equal."
    indices = np.repeat(stops - l.cumsum(), l) + np.arange(l.sum())
    return indices.reshape(-1, l[0])


def print_model_params(model):
    param_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("%-40s\t%-30s\t%-30s" % (name, list(param.shape), param.numel()))
            param_count += param.numel()
    print("%-40s\t%-30s" % ("Total trainable params", param_count))



def _line_endpoints(line):
    """Return a list of endpoints [(x1,y1),(x2,y2)] for a LineString/MultiLineString."""
    if line.is_empty:
        return []
    if isinstance(line, LineString):
        coords = list(line.coords)
        if len(coords) < 2:
            return []
        return [coords[0], coords[-1]]
    elif isinstance(line, MultiLineString):
        # take endpoints of each part (dedupe later)
        pts = []
        for part in line.geoms:
            if part.length > 0:
                c = list(part.coords)
                pts.extend([c[0], c[-1]])
        # deduplicate very-close endpoints inside a multilinestring
        if not pts:
            return []
        pts = np.array(pts)
        # simple dedupe with rounding to 1e-6 map units (safe after projecting to meters)
        uniq, idx = np.unique(np.round(pts, 6), axis=0, return_index=True)
        pts = pts[np.sort(idx)]
        return [tuple(p) for p in pts[:2]] if len(pts) >= 2 else [tuple(pts[0])]
    else:
        return []

def endpoint_adjacency(
    roads_gdf: gpd.GeoDataFrame,
    d: float = 20.0,
    crs_metric=None,
    id_col=None,
    return_sparse: bool = False,
):
    """
    Build an NxN binary adjacency where entry[i,j]=1 if any endpoint of link i
    is within distance d (meters) of any endpoint of link j.

    Parameters
    ----------
    roads_gdf : GeoDataFrame of LineString/MultiLineString
    d         : float, distance threshold in meters
    crs_metric: str or EPSG code. If provided (recommended), reprojects to this metric CRS.
                If None, assumes the GeoDataFrame is already in a metric CRS.
    id_col    : optional column with stable link IDs. If None, uses row order [0..N-1].
    return_sparse : if True, returns a scipy.sparse.csr_matrix; else returns a dense np.uint8 array.

    Returns
    -------
    A : (N,N) adjacency (csr_matrix or np.ndarray)
    index_to_id : list mapping row index -> link id (if id_col given, otherwise same as index)
    """
    # 1) project to meters if requested
    g = roads_gdf.copy()
    if crs_metric is not None:
        g = g.to_crs(crs_metric)

    # 2) establish link index <-> id
    if id_col is not None and id_col in g.columns:
        link_ids = list(g[id_col].values)
    else:
        link_ids = list(range(len(g)))
    N = len(link_ids)

    # 3) collect endpoints (2 per link, but may be fewer if geometry invalid/degenerate)
    ep_coords = []          # shape (~2N, 2)
    ep_link_idx = []        # which link this endpoint belongs to
    for i, geom in enumerate(g.geometry.values):
        if geom is None or geom.is_empty:
            continue
        pts = _line_endpoints(geom)
        # keep at most 2 endpoints per link; if only one found (degenerate), keep it
        for p in pts[:2]:
            ep_coords.append(p)
            ep_link_idx.append(i)

    if len(ep_coords) == 0:
        # no endpoints -> empty adjacency
        if return_sparse:
            from scipy.sparse import csr_matrix
            return csr_matrix((N, N), dtype=np.uint8), link_ids
        else:
            return np.zeros((N, N), dtype=np.uint8), link_ids

    ep_coords = np.asarray(ep_coords, dtype=float)
    ep_link_idx = np.asarray(ep_link_idx, dtype=int)

    # 4) radius search over endpoints
    tree = cKDTree(ep_coords)
    # For each endpoint index u, find all endpoint indices v within d
    # query_ball_point returns variable-length lists per u
    nbrs = tree.query_ball_point(ep_coords, r=d)

    # 5) fill adjacency using (link_u, link_v) from endpoint pairs
    if return_sparse:
        from scipy.sparse import coo_matrix
        rows, cols = [], []
        for u, vs in enumerate(nbrs):
            lu = ep_link_idx[u]
            for v in vs:
                lv = ep_link_idx[v]
                if lu == lv:
                    continue  # skip self
                rows.append(lu)
                cols.append(lv)
        if rows:
            data = np.ones(len(rows), dtype=np.uint8)
            A = coo_matrix((data, (rows, cols)), shape=(N, N), dtype=np.uint8).tocsr()
            # make symmetric (undirected) and zero diagonal
            A = ((A + A.T) > 0).astype(np.uint8)
            A.setdiag(0)
            A.eliminate_zeros()
        else:
            from scipy.sparse import csr_matrix
            A = csr_matrix((N, N), dtype=np.uint8)
        return A, link_ids
    else:
        A = np.zeros((N, N), dtype=np.uint8)
        for u, vs in enumerate(nbrs):
            lu = ep_link_idx[u]
            for v in vs:
                lv = ep_link_idx[v]
                if lu == lv:
                    continue
                A[lu, lv] = 1
                A[lv, lu] = 1
        np.fill_diagonal(A, 0)
        return A, link_ids
    

@torch.no_grad()
def corr_speed_posneg_from_batch(
    batch_x: torch.Tensor,               # (B, T_h, N, C) normalized
    batch_y: torch.Tensor,               # (B, T_f, N, C) raw
    scaler,                              # has .inverse_transform(tensor) -> raw
    A_mask: torch.Tensor,                # (N, N) bool; 1 = physically allowed (i->j)
    lags: Sequence[int],                 # positive lags, e.g. [1,2,3]
    tau_hi: float = 0.5,                 # significance threshold (magnitude)
    label_only: bool = False,            # True → return boolean significance only
    device: str = "cuda:0"
):
    """
    Edge-only Pearson cross-correlation on *speeds* at positive lags, split into
    positive and negative channels.

    Direction: corr(v_i(t), v_j(t+lag))  so positive lag encodes i→j timing.

    Returns (all on `device`):
      if label_only=False:
          rho_pos_edge : (B, L, E) float32   # max( rho, 0 )
          rho_neg_edge : (B, L, E) float32   # max(-rho, 0 )  (anti-corr magnitude)
          edge_index   : (2, E)   long       # rows [i_indices; j_indices]
      else:
          sig_pos_edge : (B, L, E) bool      # rho >  +tau_hi
          sig_neg_edge : (B, L, E) bool      # rho <  -tau_hi
          edge_index   : (2, E)   long
    """
    assert A_mask.dtype == torch.bool, "A_mask must be boolean (N,N)"
    assert len(lags) > 0 and all(l > 0 for l in lags), "Provide positive lags only"

    # ---- move inputs/mask to device, build directed edge list ----
    A_mask = A_mask.to(device)
    I, J = A_mask.nonzero(as_tuple=True)          # (E,)
    edge_index = torch.stack([I, J], dim=0)       # (2, E)
    E = I.numel()
    N = A_mask.shape[0]
    L = len(lags)

    # ---- recover raw speeds and stitch history+future along time ----
    bx_raw = scaler.inverse_transform(batch_x.to(device)).squeeze(-1)  # (B, T_h, N)
    by_raw = batch_y.to(device).squeeze(-1)                            # (B, T_f, N)
    X = torch.cat([bx_raw, by_raw], dim=1)                             # (B, T, N)
    B, T, _ = X.shape

    # ---- allocate outputs ----
    if label_only:
        sig_pos = torch.zeros((B, L, E), dtype=torch.bool, device=device)
        sig_neg = torch.zeros((B, L, E), dtype=torch.bool, device=device)
    else:
        rho_pos = torch.zeros((B, L, E), dtype=torch.float32, device=device)
        rho_neg = torch.zeros((B, L, E), dtype=torch.float32, device=device)

    # ---- per-lag Pearson correlation over the overlap window ----
    eps = 1e-8
    for li, lag in enumerate(lags):
        if lag >= T:
            continue

        # i at t, j at t+lag
        Xi = X[:, :T - lag, :]     # (B, Tl, N)
        Xj = X[:,  lag:   , :]     # (B, Tl, N)
        Tl = Xi.shape[1]

        # center over time per node
        Xi_c = Xi - Xi.mean(dim=1, keepdim=True)
        Xj_c = Xj - Xj.mean(dim=1, keepdim=True)

        # std over time per node
        std_i = torch.sqrt((Xi_c.pow(2)).mean(dim=1) + eps)   # (B, N)
        std_j = torch.sqrt((Xj_c.pow(2)).mean(dim=1) + eps)   # (B, N)

        # gather only allowed edges → (B, Tl, E)
        Ai = Xi_c.index_select(2, I)
        Bj = Xj_c.index_select(2, J)

        # covariance over time (B,E), denominator per edge (B,E)
        cov   = (Ai * Bj).sum(dim=1) / max(Tl - 1, 1)
        denom = (std_i.gather(1, I.expand(B, -1)) *
                 std_j.gather(1, J.expand(B, -1))).clamp_min(eps)
        rho   = cov / denom                                   # (B, E)

        if label_only:
            sig_pos[:, li, :] = (rho >  +tau_hi)
            sig_neg[:, li, :] = (rho <  -tau_hi)
        else:
            # split channels as non-negative magnitudes
            rho_pos[:, li, :] = torch.clamp(rho,  min=0.0)
            rho_neg[:, li, :] = torch.clamp(-rho, min=0.0)

    if label_only:
        return sig_pos, sig_neg, edge_index
    else:
        return rho_pos, rho_neg, edge_index
    
def bce_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Compute BCE loss between pred and target of the same shape.

    Args:
        pred:   predicted probabilities/logits in [0,1], shape (...).
        target: ground truth binary labels (0 or 1), same shape as pred.
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        torch.Tensor: scalar if reduction != 'none', else elementwise loss of same shape.
    """
    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"
    print (target)
    return torch.nn.functional.binary_cross_entropy(pred, target, reduction=reduction)
    

# def prop_graph_bce_loss(_pos_pred, _neg_pred, batch_x, batch_y, SCALER, A_mask, lags, tau_hi=0.5, label_only=True, device='cpu'):
#     _pos_edge, _neg_edge, _ = corr_speed_posneg_from_batch(
#                         batch_x,
#                         batch_y,
#                         scaler=SCALER,                              # has .inverse_transform(tensor) -> raw
#                         A_mask=A_mask.to(torch.bool),              # (N, N) bool; 1 = physically allowed (i->j)
#                         lags=lags,                 # positive lags, e.g. [1,2,3]
#                         tau_hi=tau_hi,                 # significance threshold for |rho| (if label_only)
#                         label_only=label_only,            # True → return boolean significance labels only
#                         device=device,
#                     )
#     pos_bce = bce_loss(_pos_pred, _pos_edge.to(torch.float32))
#     neg_bce = bce_loss(_neg_pred, _neg_edge.to(torch.float32))
#     return pos_bce + neg_bce

def bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute BCE loss between predicted logits and target labels.
    Args:
        logits:   predicted logits, shape (...).
        labels: ground truth binary labels (0 or 1), same shape as logits.
    Returns:
        torch.Tensor: scalar BCE loss.
    """
    assert logits.shape == labels.shape, f"Shape mismatch: {logits.shape} vs {labels.shape}"
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())

def prop_graph_bce_loss(_pos_logits, _neg_logits, batch_x, batch_y, SCALER, A_mask, lags, tau_hi=0.5, label_only=True, device='cpu'):
    _pos_edge, _neg_edge, _ = corr_speed_posneg_from_batch(
                        batch_x,
                        batch_y,
                        scaler=SCALER,                              # has .inverse_transform(tensor) -> raw
                        A_mask=A_mask.to(torch.bool),              # (N, N) bool; 1 = physically allowed (i->j)
                        lags=lags,                 # positive lags, e.g. [1,2,3]
                        tau_hi=tau_hi,                 # significance threshold for |rho| (if label_only)
                        label_only=label_only,            # True → return boolean significance labels only
                        device=device,
                    )
    pos_bce = bce_loss(_pos_logits, _pos_edge.to(torch.float32))
    neg_bce = bce_loss(_neg_logits, _neg_edge.to(torch.float32))
    return pos_bce + neg_bce