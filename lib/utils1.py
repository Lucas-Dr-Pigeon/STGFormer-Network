import json
import pygsp
import torch
import torch.utils.data as utils
import numpy as np
import pandas as pd 
import networkx as nx 
#from scipy import sparse
from sklearn.preprocessing import normalize
# import metis
from collections import defaultdict
import copy
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import os
from tqdm import trange
from .utils import print_log, StandardScaler, MinMaxScaler, vrange
from shapely.geometry import LineString, MultiLineString
from scipy.spatial import cKDTree  

def graph_reader(path):
    graph = nx.from_edgelist(pd.read_csv(path).values.tolist())
    graph.remove_edges_from(nx.selfloop_edges(graph))

    return graph

def adj_reader(path, loadtxt=False):
    adj = np.loadtxt(path) if loadtxt else np.load(path)
    ind = np.diag_indices(adj.shape[0])
    adj[ind[0],ind[1]] = torch.zeros(adj.shape[0])
    return adj

def shp_and_tmc_reader(shp_path, tmc_path):
    tmc = pd.read_csv(tmc_path, index_col=0)['0'].values
    gdf = gpd.read_file(shp_path).set_index('Tmc')
    return gdf.loc[tmc], tmc

def feature_reader(path, batch_size, seq_len, pred_len,
                   train_proportion, valid_proportion, tmc,
                   split_indices_path=None, save_split=False, log=None):
    # np.random.seed(99)
    # torch.manual_seed(99)

    df = pd.read_pickle(path).astype(float)
    feature_matrix = df.reindex(columns=tmc, fill_value=0).astype(np.float32)
    time_len = feature_matrix.shape[0]
    # print (f"-- debug df max value :{feature_matrix.values.max()}  df min value: {feature_matrix.values.min()} \
    #        feature matrix infinite?: {np.isfinite(feature_matrix.values).all()} --")

    # Build sequences
    print ("--- Building Sequences ---")

    # After: feature_matrix is (T, F) float32
    arr = feature_matrix.to_numpy(copy=True)                       # (T, F) float32
    # print (f"-- Debug arr shape:{arr.shape}--")
    T, F = arr.shape
    N = T - (seq_len + pred_len)
    # print (f"--debug predict len {pred_len}, N shape {N}--")
    if N <= 0:
        raise ValueError(f"Not enough timesteps: T={T}, seq_len={seq_len}, pred_len={pred_len}")
    
    if not np.isfinite(arr).all():
        raise ValueError("Input array has NaN/inf before sliding window")

    # Vectorized sliding windows along time
    from numpy.lib.stride_tricks import sliding_window_view
    _win = sliding_window_view(arr, window_shape=seq_len+pred_len, axis=0)
    # shape: (N, seq_len+pred_len, F)
    # print (f"-- debug windows shape {_win.shape} {seq_len, pred_len} --")
    X = _win[:, :, :seq_len]
    Y = _win[:, :, seq_len:]
    # print (f"-- debug windows shape {win.shape} feature shape {X.shape} win finite {np.isfinite(win).all()} X finite {np.isfinite(X).all()} Y finite {np.isfinite(Y).all()} --")
    # Add channel dim and make contiguous float32
    # print (f"-- debug X, Y, win shape: {X.shape} {Y.shape} {_win.shape} --")

    feature_seq   = np.expand_dims(X, axis=3)   # (N, seq_len,  F, 1)
    feature_label = np.expand_dims(Y, axis=3)    # (N, pred_len, F, 1)

    # print (f"-- debug seq value :{feature_seq.shape} {np.isfinite(feature_seq).all()} {feature_seq.max()} {feature_seq.min()} --")
    # print (f"-- debug label value :{feature_label.shape} {np.isfinite(feature_label).all()} {feature_label.max()} {feature_label.min()} --")

    # Hard sanity checks
    assert feature_seq.dtype == np.float32 and feature_label.dtype == np.float32
    if not np.isfinite(feature_seq).all() or not np.isfinite(feature_label).all():
        raise ValueError("Non-finite values in sequences/labels")

    # 2) Sanity checks
    if not np.isfinite(feature_seq).all():
        bad = np.sum(~np.isfinite(feature_seq))
        raise ValueError(f"Found {bad} non-finite values in feature_seq")

    sample_size = feature_seq.shape[0]

    # --- Load or create consistent split indices ---
    if split_indices_path and os.path.exists(split_indices_path):
        split_data = np.load(split_indices_path)
        print (f"Load split from {split_indices_path}")
        index = split_data['index']
    else:
        index = np.arange(sample_size, dtype=int)
        np.random.shuffle(index)
        if split_indices_path and save_split:
            if not os.path.exists(split_indices_path):
                print (f"New split saved to {split_indices_path}")
                np.savez(split_indices_path, index=index)

    # Split
    train_index = int(np.floor(sample_size * train_proportion))
    valid_index = int(np.floor(sample_size * (train_proportion + valid_proportion)))

    train_data, train_label = feature_seq[index[:train_index]], feature_label[index[:train_index]]
    valid_data, valid_label = feature_seq[index[train_index:valid_index]], feature_label[index[train_index:valid_index]]
    test_data, test_label = feature_seq[index[valid_index:]], feature_label[index[valid_index:]]

    # print (f"-- debug train data finite {np.isfinite(train_data).all()} {np.isfinite(train_label).all()} {train_data.min()} {train_data.max()} --")
    # print (f"-- debug valid data finite {np.isfinite(valid_data).all()} {np.isfinite(valid_label).all()} {valid_data.min()} {valid_data.max()} --")
    # print (f"-- debug test data finite {np.isfinite(test_data).all()} {np.isfinite(test_label).all()} {train_data.min()} {test_data.max()} --")

    # print (f"-- train mean:{train_data.mean()}, train std:{train_data.std()}, train max:{train_data.max()}, train min:{train_data.min()} --")
    # print (f"-- valid mean:{valid_data.mean()}, valid std:{valid_data.std()}, valid max:{valid_data.max()}, valid min:{valid_data.min()} --")
    # print (f"-- test mean:{test_data.mean()}, test std:{test_data.std()}, test max:{test_data.max()}, test min:{test_data.min()} --")

    print ("--- Scaling Sequences ---")
    scaler = MinMaxScaler(min_val=0, max_val=99, min_feature=0, max_feature=1)

    train_data = scaler.transform(train_data)
    valid_data = scaler.transform(valid_data)
    test_data = scaler.transform(test_data)

    print_log(f"Trainset:\tx-{train_data.shape}\ty-{train_label.shape}", log=log)
    print_log(f"Valset:  \tx-{valid_data.shape}  \ty-{valid_label.shape}", log=log)
    print_log(f"Testset:\tx-{test_data.shape}\ty-{test_label.shape}", log=log)

    # Convert to tensors
    train_dataset = utils.TensorDataset(torch.Tensor(train_data).permute(0,2,1,3), torch.Tensor(train_label).permute(0,2,1,3))
    valid_dataset = utils.TensorDataset(torch.Tensor(valid_data).permute(0,2,1,3), torch.Tensor(valid_label).permute(0,2,1,3))
    test_dataset = utils.TensorDataset(torch.Tensor(test_data).permute(0,2,1,3), torch.Tensor(test_label).permute(0,2,1,3))

    # Create dataloaders
    train_dataloader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dataloader = utils.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_dataloader, valid_dataloader, test_dataloader, scaler 


def save_logs(args, logs):
    """
    Save the logs at the path.
    :param args: Arguments objects.
    :param logs: Log dictionary.
    """
    with open(args.log_path, "w") as f:
        json.dump(logs, f)

class SubgraphMaker:
    @staticmethod
    def split_into_islands(graph):
        components = list(nx.connected_components(graph))
        islands = [graph.subgraph(c).copy() for c in components]
        return islands
    
    @staticmethod
    def metis_partition_subgraph(subgraph, num_parts=30):
        edgecuts, parts = metis.part_graph(subgraph, nparts=num_parts)
        partition_map = defaultdict(list)
        for node, part in zip(subgraph.nodes(), parts):
            partition_map[part].append(node)
        return partition_map
    
    @staticmethod
    def merge_small_regions(df_mapping, min_nodes=10):
        # Step 1: Initialize maps
        region_to_nodes = defaultdict(set)
        node_to_island = {}

        for row in df_mapping.itertuples():
            region_to_nodes[row.region_id].add(row.node)
            node_to_island[row.node] = row.island_id

        # Step 2: Separate small vs. large regions
        small_regions = {r: nodes for r, nodes in region_to_nodes.items() if len(nodes) < min_nodes}
        large_regions = {r: nodes for r, nodes in region_to_nodes.items() if len(nodes) >= min_nodes}

        # Step 3: Merge all small regions
        merged_regions = []
        new_region_id = 0
        small_nodes_flat = [n for nodes in small_regions.values() for n in nodes]

        if len(small_nodes_flat) >= min_nodes:
            merged_regions.append((new_region_id, small_nodes_flat))
            new_region_id += 1
        else:
            # Find the smallest large region
            if large_regions:
                target_region = min(large_regions.items(), key=lambda x: len(x[1]))[0]
                region_to_nodes[target_region].update(small_nodes_flat)
            else:
                # All regions are small, merge everything
                merged_regions.append((new_region_id, small_nodes_flat))
                new_region_id += 1

        # Step 4: Add large regions as-is (except target_region if already merged into)
        skip_region = target_region if len(small_nodes_flat) < min_nodes else None

        for r, nodes in region_to_nodes.items():
            if r in small_regions or r == skip_region:
                continue
            merged_regions.append((new_region_id, list(nodes)))
            new_region_id += 1

        if len(small_nodes_flat) < min_nodes and skip_region is not None:
            merged_regions.append((new_region_id, list(region_to_nodes[skip_region])))
            new_region_id += 1

        # Step 5: Build final df_mapping
        final_records = []
        for region_id, nodes in merged_regions:
            for node in nodes:
                final_records.append({
                    'node': node,
                    'region_id': region_id,
                    'island_id': node_to_island[node]
                })

        return pd.DataFrame(final_records).sort_values("node").reset_index(drop=True)
    
    @staticmethod
    def get_linestring_coordinates(gdf, prj=3857):
        # Automatically convert to metric CRS (e.g., EPSG:3857)
        links_metric = gdf.to_crs(epsg=prj)
        # Step 2: Extract start and end points
        start_coords = links_metric.geometry.apply(lambda geom: geom.coords[0])
        end_coords = links_metric.geometry.apply(lambda geom: geom.coords[-1])
        # Step 3: Create separate x/y columns for start and end
        links_metric['start_x'] = [pt[0] for pt in start_coords]
        links_metric['start_y'] = [pt[1] for pt in start_coords]
        links_metric['end_x'] = [pt[0] for pt in end_coords]
        links_metric['end_y'] = [pt[1] for pt in end_coords]
        # Step 4: Format as coords_df
        coords_df = links_metric[['start_x', 'start_y', 'end_x', 'end_y']].copy()
        coords_df['tmc'] = coords_df.index
        coords_df['node'] = range(len(coords_df))
        coords_df = coords_df[['node','tmc', 'start_x', 'start_y', 'end_x', 'end_y']].reset_index(drop=True)
        return coords_df

    @staticmethod
    def get_partitioned_regions(adj, gdf, expected_num_parts=100, min_cut=30, vis=True):
        G = nx.from_numpy_array(adj)
        # Get coords_Df
        coords_df =  SubgraphMaker.get_linestring_coordinates(gdf)
        # Get connected islands
        islands = SubgraphMaker.split_into_islands(G)
        node_records = []
        # Perform METIS cutting within each island
        for island in islands:
            if island.number_of_nodes() >= expected_num_parts:
                num_parts = max(2, island.number_of_nodes() // expected_num_parts)
                partition_map = SubgraphMaker.metis_partition_subgraph(island, num_parts)
                for part_nodes in partition_map.values():
                    node_records.append(part_nodes)
            else:
                node_records.append(list(island.nodes))

        small_cuts = [ reg for reg in node_records if len(reg)<min_cut  ]
        large_cuts = [ reg for reg in node_records if len(reg)>=min_cut  ]

        records = []
        for region_id, nodes in enumerate(large_cuts):
            for node in nodes:
                records.append({"node": node, "region_id": region_id})

        # Create DataFrame
        df_mapping = pd.DataFrame(records).set_index('node').sort_index()
        large_nodes = np.concatenate(large_cuts)
        large_cut_nodes = coords_df.loc[large_nodes]
        l1 = torch.tensor(large_cut_nodes[['start_x','start_y']].values, device='cuda')
        l2 = torch.tensor(large_cut_nodes[['end_x','end_y']].values, device='cuda')

        #
        small_assigned_to = np.zeros(len(small_cuts))
        regions = copy.deepcopy(large_cuts)
        for sr, sregion in enumerate(small_cuts):
            small_cut_nodes = coords_df.loc[sregion]
            s1 = torch.tensor(small_cut_nodes[['start_x','start_y']].values, device='cuda')
            s2 = torch.tensor(small_cut_nodes[['end_x','end_y']].values, device='cuda')

            d1 = torch.cdist(s1, l1)
            d2 = torch.cdist(s1, l2)
            d3 = torch.cdist(s2, l1)
            d4 = torch.cdist(s2, l2)

            dists = torch.stack([d1, d2, d3, d4], dim=0)
            flat_closest_ix = torch.argmin(dists)
            _, _M, _F = dists.shape
            n = flat_closest_ix // (_M * _F)
            m = (flat_closest_ix % (_M * _F)) // _F
            f = flat_closest_ix % _F
            min_index = (n.item(), m.item(), f.item())
            closest_ix =  min_index[-1]
            closest_large_node = large_nodes[closest_ix]
            assigned_region = df_mapping.loc[closest_large_node].values[0]
            small_assigned_to[sr] = assigned_region
            _ = [ regions[assigned_region].append(_node) for _node in sregion ]

        if vis:
            # Total number of regions
            n_regions = len(regions)

            # Create a color map
            cmap = cm.get_cmap('tab20', n_regions)
            norm = colors.Normalize(vmin=0, vmax=n_regions - 1)

            # Prepare a plot
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.set_title("Road Links Colored by Region")
            ax.axis('off')

            # Plot each region with a different color
            for region_idx, node_indices in enumerate(regions):
                subset = gdf.iloc[node_indices]
                subset.plot(ax=ax, color=cmap(region_idx), linewidth=1, label=f'Region {region_idx}')

            # Optional: Add legend
            ax.legend()
            plt.show()

        return regions


class GetWavelet(object):
    """docstring for GetWavelet"""
    def __init__(self, adj, scale, approximation_order, tolerance, scale_offsets=[0, 3, 5]):
        super(GetWavelet, self).__init__()
        self.adj = adj
        self.pygsp_graph = pygsp.graphs.Graph(self.adj)
        self.pygsp_graph.estimate_lmax()
        self.pygsp_graph.compute_laplacian('combinatorial')
        self.scales = np.concatenate([[-scale-offset, scale+offset] for offset in scale_offsets ])
        self.approximation_order = approximation_order
        self.tolerance = tolerance
        self.phi_matrices = []

    def calculate_wavelet(self):
        impulse = np.eye(self.adj.shape[0], dtype=int)
        wavelet_coefficients = pygsp.filters.approximations.cheby_op(self.pygsp_graph, self.chebyshev, impulse)
        wavelet_coefficients[wavelet_coefficients<self.tolerance] = 0

        return wavelet_coefficients

    def normalize_matrices(self):

        print("\nNormalizing the wavelets.\n")
        for i, phi_matrix in enumerate(self.phi_matrices):
            self.phi_matrices[i] = normalize(self.phi_matrices[i], norm='l1', axis=1)

    def calculate_all_wavelets(self):

        print(f"\nWavelet calculation started. Scales:{self.scales} \n")
        for i, scale in enumerate(self.scales):
            self.heat_filter = pygsp.filters.Heat(self.pygsp_graph, tau=[scale])
            self.chebyshev = pygsp.filters.approximations.compute_cheby_coeff(self.heat_filter, m=self.approximation_order)
            calculated_wavelets = self.calculate_wavelet()
            self.phi_matrices.append(calculated_wavelets)
        self.phi_matrices = np.asarray(self.phi_matrices)
        self.normalize_matrices()

def get_region_wavelets(adj, subgraph_mapping, scale, approximation_order, tolerance, scale_offsets):
    '''
    subgraph_mapping: {region_id: [nodes]}
    '''
    regional_wavelets = {}
    for region_id in range(len(subgraph_mapping)):
        nodes = subgraph_mapping[region_id]
        # Subgraph adjacency
        A_r = adj[np.ix_(nodes, nodes)]  # submatrix for this region

        # Compute wavelet for this region
        wavelet_r = GetWavelet(adj=A_r, scale=scale, approximation_order=approximation_order, tolerance=tolerance, scale_offsets=scale_offsets)
        wavelet_r.calculate_all_wavelets()  # stores result in wavelet_r.phi_matrices

        regional_wavelets[region_id] = {
            'nodes': nodes,  # global indices
            'wavelet': wavelet_r,  # shape: (num_scales, N_r, N_r)
        }
    return regional_wavelets



        
        