import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from typing import Dict
import time
import warnings
warnings.filterwarnings('ignore')
import copy
import pickle
from scipy.spatial import KDTree
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from . import logger
from tokenizer import GeneVocab
from tokenizer import tokenize_and_pad_batch, random_mask_value
from model import TransformerModel
from loss import masked_mse_loss
from sklearn.preprocessing import LabelEncoder
import random
import pickle
import torch.nn.functional as F
import concurrent.futures
from collections import defaultdict
from tqdm import tqdm

def convert_gene_symbols_to_ids(adata, tokenizer_dir):
    scfoundation_gene_df = pd.read_csv(f'{tokenizer_dir}/scfoundation_gene_df.csv')
    scfoundation_gene_df = scfoundation_gene_df.drop_duplicates(subset='gene_symbols', keep='first')
    scfoundation_gene_df['gene_symbols'] = scfoundation_gene_df['gene_symbols'].str.lower()
    scfoundation_gene_df.set_index('gene_symbols', inplace=True)
    gene_symbols_in_adata = adata.var_names.str.lower()

    valid_gene_symbols = np.intersect1d(gene_symbols_in_adata, scfoundation_gene_df.index)
    gene_ids = scfoundation_gene_df.loc[valid_gene_symbols, 'gene_ids'].values
    valid_indices = np.isin(adata.var_names.str.lower(), valid_gene_symbols)
    adata = adata[:, valid_indices]
    adata.var_names = np.array(gene_ids, dtype=str)

    return adata

def read_P_adata(path, time_point):
    adata = sc.read_h5ad(path+f'{time_point}_xenium.h5ad')
    adata.X = adata.X.astype(np.float32)
    ### spatial info
    coord = adata.obs[['coord_x','coord_y']].values
    coord = coord.astype(np.float32)
    adata.obsm['spatial'] = coord

    return adata

class SlideData():
    def __init__(self, data_path, slide, vocab, mask_ratio, mask_value, pad_value, pad_token, time_points=None, save_hvg=False, load_hvg=False):
        self.data_path = data_path
        self.slide = slide
        self.vocab = vocab
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value
        self.pad_value = pad_value
        self.pad_token = pad_token
        self.save_hvg = save_hvg
        self.load_hvg = load_hvg
        # Define the full list of time_points (default list)
        full_time_points = [
            'P0_1', 'P0_2',
            'P5_1', 'P5_2',
            'P10_1', 'P10_2',
            'P14_1', 'P14_2',
            'P28_1', 'P28_2',
            'P56_1', 'P56_2',
            'E115_LMGE_1', 'E115_LMGE_2',
            'E125_LMGE_1', 'E125_LMGE_2','E125_LMGE_3',
            'E135_LMGE_1',#'E135_LMGE_2',
            'E145_LMGE_1', 'E145_LMGE_2', 'E145_LMGE_3',
            'E155_LMGE_1', #'E155_LMGE_2', 
            'E115_CGE_1', 'E115_CGE_2',
            'E125_CGE_1', 'E125_CGE_2',
            'E135_CGE_1',#'E135_CGE_2',
            'E145_CGE_1',#'E145_CGE_2',
            'E155_CGE_1', #'E155_CGE_2', 
        ]
        # If time_points is None, use the full list; else, use the provided subset
        if time_points is None:
            time_points = full_time_points.copy()
        self.time_points = time_points

        # Fit LabelEncoders on the full list of batches and times
        self.batch_encoder = LabelEncoder().fit(full_time_points)  # Each time_point is a batch
        full_times = [tp.split('_')[0] for tp in full_time_points]
        self.time_encoder = LabelEncoder().fit(full_times)
        self.load_data(time_points)

    def load_data(self, time_points):
        # Parallel data loading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_timepoint = {
                executor.submit(read_P_adata, self.data_path, time_point): time_point 
                for time_point in time_points
            }
            adata_dict = {
                future_to_timepoint[future]: future.result() 
                for future in concurrent.futures.as_completed(future_to_timepoint)
            }

        var_names = adata_dict[time_points[0]].var_names
        for time_point in time_points[1:]:
            var_names = var_names.intersection(adata_dict[time_point].var_names)

        for time_point in time_points:
            adata_dict[time_point] = adata_dict[time_point][:, var_names]
        
        adata = sc.concat(list(adata_dict.values()), axis=0)

        if self.save_hvg:
            sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=3000, batch_key='sample_info', )
            # Save HVG
            with open('hvg_10Xdata_CLTV2.pkl', 'wb') as file:
                pickle.dump(adata.var['highly_variable'], file)
            adata = adata[:, adata.var['highly_variable']]

        if self.load_hvg:
            # Load HVG
            with open('hvg_10Xdata_CLTV2.pkl', 'rb') as file:
                HVGs = pickle.load(file)
            adata = adata[:, adata.var_names.isin(HVGs.index)]
            adata = adata[:, HVGs]

        adata = convert_gene_symbols_to_ids(adata, tokenizer_dir)

        scfoundation_gene_df = pd.read_csv(f'{tokenizer_dir}/scfoundation_gene_df.csv')
        scfoundation_gene_df.set_index('gene_ids', inplace=True)
        total_gene_num = adata.shape[1]
        adata = adata[:, adata.var_names.isin(scfoundation_gene_df.index)]
        adata.var['gene_name'] = scfoundation_gene_df.loc[adata.var_names, 'gene_symbols'].values
        seleted_gene_num = adata.shape[1]
        genes = adata.var["gene_name"].tolist()
        gene_ids = np.array(self.vocab(genes), dtype=int)


        if self.save_hvg:
            sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=1000, batch_key='sample_info', )
            # Save HVG
            with open('hvg_10Xdata_niche_CLTV2.pkl', 'wb') as file:
                pickle.dump(adata.var['highly_variable'], file)
            highly_variable_gene_names = adata.var['gene_name'][adata.var['highly_variable']].tolist()

        if self.load_hvg:
            # Load HVG
            with open('hvg_10Xdata_niche_CLTV2.pkl', 'rb') as file:
                HVGs = pickle.load(file)
            highly_variable_gene_names = adata[:, adata.var_names.isin(HVGs.index)].var['gene_name'][HVGs].tolist()
        
        highly_variable_gene_names = adata.var['gene_name'].tolist()
        hvg_ids = np.array(self.vocab(highly_variable_gene_names), dtype=int)
        # print(len(highly_variable_gene_names))

        logger.info(
            f"match {seleted_gene_num}/{total_gene_num} genes "
            f"in vocabulary of size 19264."
        )

        le = LabelEncoder()
        self.adata = adata
        self.gene_ids = gene_ids
        self.ligand_ids = hvg_ids
        self.xdata = self.adata.X.astype(np.float32)
        self.spatial = self.adata.obsm['ccf_l']
        self.tree = KDTree(self.spatial)
        self.celltype = le.fit_transform(adata.obs['CellType'])


        # Encode batch and time using pre-fitted encoders
        loaded_batches = [batch[0].upper()+batch[1:] for batch in adata.obs['batch'].values]
        # Convert batches to ippercase before encoding
        self.batch = self.batch_encoder.transform(loaded_batches)
        loaded_times = [t.split('_')[0] for t in loaded_batches]
        self.time = self.time_encoder.transform(loaded_times)
        self.batch_name = loaded_batches#np.array(loaded_batches)

    def get_niche_samples(self):
        samples_expression = []
        samples_ctprop = []

        for i in range(self.adata.shape[0]):
            _, knn_data_index = self.tree.query(self.spatial[i], k=3)
            niche_counts = np.concatenate([self.xdata[knn_data_index[1]].A, self.xdata[knn_data_index[2]].A])
            niche_counts[:,~np.isin(self.gene_ids, self.ligand_ids)] = 0
            samples_expression.append(np.concatenate([self.xdata[i].A, niche_counts],axis=0))

            weight_list = []
            for sub_i in range(1,3):
                if self.celltype[knn_data_index[sub_i]] == self.celltype[i]:
                    weight_list.append(3)
                else:
                    weight_list.append(1)
            weight_array = np.array(weight_list) / np.sum(np.array(weight_list))

            samples_ctprop.append(weight_array)
        self.expression = samples_expression
        self.ctprop = samples_ctprop

    def tokenize_data(self):
        (train_data,
         valid_data,
         train_ctprop,
         valid_ctprop,
         train_celltype,
         valid_celltype,
         train_spatial,
         valid_spatial,
         train_batch,
         valid_batch,
         train_time,
         valid_time,
         train_batch_name,
         valid_batch_name,
         ) = train_test_split(
            self.expression, self.ctprop, self.celltype, self.spatial, self.batch, self.time, self.batch_name, test_size=0.1, shuffle=True
        )

        max_seq_len = np.max(np.count_nonzero(self.adata.X.A, axis=1))+2
        max_niche_cell_num = 4
        self.max_seq_len = max_seq_len
        self.max_niche_cell_num = max_niche_cell_num

        tokenized_train = tokenize_and_pad_batch(
            train_data,
            train_ctprop,
            self.gene_ids,
            max_len = max_seq_len,
            max_niche_cell_num = max_niche_cell_num,
            vocab = self.vocab,
            pad_token = self.pad_token,
            pad_value = self.pad_value,
            append_cls = False,
            include_zero_gene = False,
        )

        tokenized_valid = tokenize_and_pad_batch(
            valid_data,
            valid_ctprop,
            self.gene_ids,
            max_len = max_seq_len,
            max_niche_cell_num = max_niche_cell_num,
            vocab = self.vocab,
            pad_token = self.pad_token,
            pad_value = self.pad_value,
            append_cls = False,
            include_zero_gene = False,
        )

        logger.info(
            f"train set number of samples: {tokenized_train['center_genes'].shape[0]}, "
            f"\n\t feature length of center cell: {tokenized_train['center_genes'].shape[1]}"
            f"\n\t feature length of niche cells: {tokenized_train['niche_genes'].shape[1]}"
        )
        logger.info(
            f"valid set number of samples: {tokenized_valid['center_genes'].shape[0]}, "
            f"\n\t feature length of center cell: {tokenized_valid['center_genes'].shape[1]}"
            f"\n\t feature length of niche cells: {tokenized_valid['niche_genes'].shape[1]}"
        )

        self.tokenized_train = tokenized_train
        self.tokenized_valid = tokenized_valid
        ## add
        self.train_celltype = train_celltype
        self.valid_celltype = valid_celltype
        self.train_spatial = train_spatial
        self.valid_spatial = valid_spatial
        self.train_batch = train_batch
        self.valid_batch = valid_batch
        self.train_time = torch.from_numpy(train_time).long()
        self.valid_time = torch.from_numpy(valid_time).long()
        self.train_batch_name = train_batch_name
        self.valid_batch_name = valid_batch_name

    def prepare_data(self):
        masked_values_train = random_mask_value(
            self.tokenized_train["center_values"],
            mask_ratio = self.mask_ratio,
            mask_value = self.mask_value,
            pad_value = self.pad_value,
        )
        masked_values_valid = random_mask_value(
            self.tokenized_valid["center_values"],
            mask_ratio = self.mask_ratio,
            mask_value = self.mask_value,
            pad_value = self.pad_value,
        )
        logger.info(
            f"random masking ratio of masked values in train: "
            f"{(masked_values_train == self.mask_value).sum() / (masked_values_train - self.pad_value).count_nonzero() *100:2.2f}%"
            f"\n\t\t  random masking ratio of masked values in valid: "
            f"{(masked_values_valid == self.mask_value).sum() / (masked_values_valid - self.pad_value).count_nonzero() *100:2.2f}%"
        )

        train_data_pt = {
            "center_gene_ids": self.tokenized_train["center_genes"],
            "input_center_values": masked_values_train,
            "target_center_values": self.tokenized_train["center_values"],
            "niche_gene_ids": self.tokenized_train["niche_genes"],
            "input_niche_values": self.tokenized_train["niche_values"],
            "niche_feature_lens": self.tokenized_train["niche_feature_lens"],
            "cross_attn_bias": self.tokenized_train["cross_attn_bias"],
            "celltype": self.train_celltype,
            "spatial": self.train_spatial,
            "batch": self.train_batch,
            "time": self.train_time,
            # "batch_name": self.train_batch_name,
            "batch_name": np.array(self.train_batch_name),
        }

        valid_data_pt = {
            "center_gene_ids": self.tokenized_valid["center_genes"],
            "input_center_values": masked_values_valid,
            "target_center_values": self.tokenized_valid["center_values"],
            "niche_gene_ids": self.tokenized_valid["niche_genes"],
            "input_niche_values": self.tokenized_valid["niche_values"],
            "niche_feature_lens": self.tokenized_valid["niche_feature_lens"],
            "cross_attn_bias": self.tokenized_valid["cross_attn_bias"],
            "celltype": self.valid_celltype,
            "spatial": self.valid_spatial,
            "batch": self.valid_batch,
            "time": self.valid_time,
            # "batch_name": self.valid_batch_name,
            "batch_name": np.array(self.valid_batch_name),
        }

        return train_data_pt, valid_data_pt

class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["center_gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

def is_pair(batch1, batch2):
    period_dict = {
        'P0': ['E155','P5'],
        'P5': ['P0', 'P10'],
        'P10': ['P5', 'P14'],
        'P14': ['P10', 'P28'],
        'P28': ['P14', 'P56'],
        'P56': ['P28'],
        ###
        'E115': ['E125'],
        'E125': ['E115', 'E135'],
        'E135': ['E125', 'E145'],
        'E145': ['E135', 'E155'],
        'E155': ['E145', 'P0'],
    }
    if batch2.split('_')[0] in period_dict[batch1.split('_')[0]]:
        return True
    else:
        return False

class SeqDataset_pair(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor], spatial_threshold: float = 0.1, k_neighbors: int = 20):
        self.data = data
        # self.batch = data["batch_name"]
        self.celltype = data["celltype"]
        self.spatial = data["spatial"]
        self.spatial_threshold = spatial_threshold
        self.k_neighbors = k_neighbors
        
        self.valid_pairs = self._compute_valid_pairs()
    
    def _compute_valid_pairs(self):
        valid_pairs = []
        unique_batches = np.unique(self.data["batch_name"])
        batch_indices = {b: np.where(self.data["batch_name"] == b)[0] for b in unique_batches}
        # Build KDTree for each batch
        trees = {b: KDTree(self.spatial[indices]) for b, indices in batch_indices.items()}
        
        # For each batch pair, find close points
        for batch1 in unique_batches:
            for batch2 in unique_batches:
                if (not is_pair(batch1, batch2)) and (batch1[:-2] != batch2[:-2]):  # Skip same batch and avoid duplicate pairs
                    continue
                indices1 = batch_indices[batch1]
                indices2 = batch_indices[batch2]

                # Query k nearest neighbors
                distances, neighbors = trees[batch2].query(
                    self.spatial[indices1], 
                    k=100, 
                )
                
                # For each point in batch1
                for i, (dist, neighs) in enumerate(zip(distances, neighbors)):
                    valid_neighs = neighs[dist != np.inf]  # Filter out points beyond threshold
                    if len(valid_neighs) == 0:
                        continue
                        
                    idx1 = indices1[i]
                    ct1 = self.celltype[idx1]
                    
                    # Find neighbors with same celltype
                    valid_pairs_temp = []
                    for n in valid_neighs:
                        idx2 = indices2[n]
                        if self.celltype[idx2] == ct1:
                            valid_pairs_temp.append((idx1, idx2))
                            
                    if len(valid_pairs_temp) >= 5:
                        valid_pairs.extend(random.sample(valid_pairs_temp, k=5))
                    else:
                        # Handle cases with fewer than 5 elements, e.g., append all or skip
                        valid_pairs.extend(valid_pairs_temp)  # Append all if acceptable
        
        return valid_pairs

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        idx1, idx2 = self.valid_pairs[idx]
        
        # Create copies of the data dictionaries
        sample1 = {k: v[idx1] if isinstance(v, (torch.Tensor, np.ndarray)) else v 
                for k, v in self.data.items() if k != 'batch_name'}
        sample2 = {k: v[idx2] if isinstance(v, (torch.Tensor, np.ndarray)) else v 
                for k, v in self.data.items() if k != 'batch_name'}
        
        # Handle batch_name separately
        sample1['batch_name'] = self.data['batch_name'][idx1]
        sample2['batch_name'] = self.data['batch_name'][idx2]
        
        return {'sample1': sample1, 'sample2': sample2}

def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    if num_workers == 0:
        num_workers = len(os.sched_getaffinity(0))

    dataset = SeqDataset_pair(
            data_pt,
            )

    def custom_collate_fn(batch):
        # Initialize dictionaries for sample1 and sample2
        collated_batch = {
            'sample1': {},
            'sample2': {}
        }
        
        # Get all keys from the first batch
        sample_keys = batch[0]['sample1'].keys()
        
        for key in sample_keys:
            if key == 'batch_name':
                # Handle batch_name separately - keep as list of strings
                collated_batch['sample1'][key] = [item['sample1'][key] for item in batch]
                collated_batch['sample2'][key] = [item['sample2'][key] for item in batch]
            else:
                # Handle other data normally
                collated_batch['sample1'][key] = torch.stack([torch.tensor(item['sample1'][key]) for item in batch])
                collated_batch['sample2'][key] = torch.stack([torch.tensor(item['sample2'][key]) for item in batch])
        
        return collated_batch

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,
        collate_fn=custom_collate_fn,  # Add this line
    )
    return data_loader

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def preprocess_on_gpu(sample, device):
    niche_feature_lens = sample["niche_feature_lens"].to(device)
    center_gene_ids = sample["center_gene_ids"].to(device)
    input_center_values = sample["input_center_values"].to(device)
    target_center_values = sample["target_center_values"].to(device)
    niche_gene_ids = sample["niche_gene_ids"].to(device)
    input_niche_values = sample["input_niche_values"].to(device)
    cross_attn_bias = sample["cross_attn_bias"].to(device)
    batch_time = sample["time"].to(device)

    encoder_src_key_padding_mask = niche_gene_ids.eq(vocab[pad_token])
    decoder_src_key_padding_mask = center_gene_ids.eq(vocab[pad_token])
    decoder_masked_positions = input_center_values.eq(mask_value)

    return {
        "niche_feature_lens": niche_feature_lens,
        "center_gene_ids": center_gene_ids,
        "input_center_values": input_center_values,
        "target_center_values": target_center_values,
        "niche_gene_ids": niche_gene_ids,
        "input_niche_values": input_niche_values,
        "cross_attn_bias": cross_attn_bias,
        "batch_time": batch_time,
        "encoder_src_key_padding_mask": encoder_src_key_padding_mask,
        "decoder_src_key_padding_mask": decoder_src_key_padding_mask,
        "decoder_masked_positions": decoder_masked_positions,
    }

def train(model: nn.Module, loader: DataLoader, epoch, slide_num, slide, mse_train) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_mse = 0.0
    total_ctr = 0.0

    start_time = time.time()

    num_batches = len(loader)

    for batch, batch_data in enumerate(loader):
        sample1 = batch_data['sample1']
        sample2 = batch_data['sample2']

        # Preprocess on GPU
        sample1 = preprocess_on_gpu(sample1, device)
        sample2 = preprocess_on_gpu(sample2, device)

        # mse loss
        with torch.cuda.amp.autocast(enabled=amp):

            output_dict1 = model(
                sample1["niche_gene_ids"],
                sample1["input_niche_values"],
                sample1["encoder_src_key_padding_mask"],
                sample1["center_gene_ids"],
                sample1["input_center_values"],
                sample1["decoder_src_key_padding_mask"],
                sample1["cross_attn_bias"],
                batch_labels=sample1["batch_time"],
                )
            loss_mse1 = criterion(
                output_dict1["mlm_output"], sample1["target_center_values"], sample1["decoder_masked_positions"]
            )

            output_dict2 = model(
                sample2["niche_gene_ids"],
                sample2["input_niche_values"],
                sample2["encoder_src_key_padding_mask"],
                sample2["center_gene_ids"],
                sample2["input_center_values"],
                sample2["decoder_src_key_padding_mask"],
                sample2["cross_attn_bias"],
                batch_labels=sample2["batch_time"],
                )
            loss_mse2 = criterion(
                output_dict2["mlm_output"], sample2["target_center_values"], sample2["decoder_masked_positions"]
            )

            loss_mse = 0.5 * loss_mse1 + 0.5 * loss_mse2
            # contrastive loss
            loss_ctr1 = loss_fn(output_dict1["cell_emb_m"], output_dict2["cell_emb"].detach())
            loss_ctr2 = loss_fn(output_dict2["cell_emb_m"], output_dict1["cell_emb"].detach())
            loss_ctr =  loss_ctr1 + loss_ctr2
            loss_ctr = loss_ctr.mean()

        model.zero_grad()
        scaler.scale(loss_mse + loss_ctr).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        total_mse += loss_mse.item()
        total_ctr += loss_ctr.item()

        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            sec_per_batch = (time.time() - start_time) / log_interval
            cur_mse = total_mse / log_interval
            cur_ctr = total_ctr / log_interval
            logger.info(
                f"| {dataset} epoch {epoch:2d} - slide {slide_num:2d} {slide} | "
                f"{batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.8f} | sec/batch {sec_per_batch:5.1f} | "
                f"mse {cur_mse:5.5f} | "
                f"ctr {cur_ctr:5.5f} | "
            )
            mse_train.append(cur_mse)
            total_mse = 0
            total_ctr = 0
            start_time = time.time()


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_num = 0
    print('valid_loder_lenth: ', len(loader))
    with torch.no_grad():
        for batch_data in tqdm(loader):
            sample1 = batch_data['sample1']
            # Preprocess on GPU
            sample1 = preprocess_on_gpu(sample1, device)

            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = model(
                    sample1["niche_gene_ids"],
                    sample1["input_niche_values"],
                    sample1["encoder_src_key_padding_mask"],
                    sample1["center_gene_ids"],
                    sample1["input_center_values"],
                    sample1["decoder_src_key_padding_mask"],
                    sample1["cross_attn_bias"],
                    batch_labels=sample1["batch_time"],
                    )
                loss = criterion(output_dict["mlm_output"], sample1["target_center_values"], sample1["decoder_masked_positions"])
            
            total_loss += loss.item() * sample1["decoder_masked_positions"].sum().item()
            total_num += sample1["decoder_masked_positions"].sum().item()
    
    return total_loss / total_num
        

def train_and_evaluate(model, train_data_pt, valid_data_pt, epoch, batch_size, slide_num, slide, mse_train, mse_valid):
    best_val_loss = float("inf")
    start_time = time.time()

    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size,
        shuffle = False,
        drop_last = True,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size,
        shuffle = False,
        drop_last = True,
    )

    train(model, train_loader, epoch, slide_num, slide, mse_train)

    val_loss = evaluate(model, valid_loader)
    mse_valid.append(val_loss)
        
    elapsed = time.time() - start_time
        
    logger.info("-" * 89)
    logger.info(
        f"| end of {dataset} epoch {epoch:2d} - slide {slide_num:2d} {slide} | "
        f"time: {elapsed:5.2f}s | "
        f"valid loss/mse {val_loss:5.4f}"
    )
    logger.info("-" * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        logger.info(f"Best model with score {best_val_loss:5.4f}")


### Prepare the pretraining model
if __name__ == '__main__':
    time_points = [
                'P0_1', 'P0_2',
                'P5_1', 'P5_2',
                'P10_1', 'P10_2',
                'P14_1', 'P14_2',
                'P28_1', 'P28_2',
                'P56_1', 'P56_2',
                'E115_LMGE_1', 'E115_LMGE_2',
                'E125_LMGE_1', 'E125_LMGE_2','E125_LMGE_3',
                'E135_LMGE_1',#'E135_LMGE_2',
                'E145_LMGE_1', 'E145_LMGE_2', 'E145_LMGE_3',
                'E155_LMGE_1', #'E155_LMGE_2', 
                'E115_CGE_1', 'E115_CGE_2',
                'E125_CGE_1', 'E125_CGE_2',
                'E135_CGE_1',#'E135_CGE_2',
                'E145_CGE_1',#'E145_CGE_2',
                'E155_CGE_1', #'E155_CGE_2', 
            ]
    num_times = np.unique([time_point.split('_')[0] for time_point in time_points]).shape[0]

    logger.info("Initialize pretraining model")
    embsize = 768
    d_hid = 3072
    nhead = 12
    nlayers = 6
    dropout = 0.1
    cell_emb_style = 'max-pool'

    logger.info("Loading scFoundation model ...")
    from tasks.scfoundation import load
    pretrainmodel, pretrainconfig = load.load_model_frommmf('models/models.ckpt')

    model = TransformerModel(
        embsize,
        nhead,
        d_hid,
        nlayers,
        dropout = dropout,
        do_dab=False,
        cell_emb_style = cell_emb_style,
        scfoundation_token_emb1 = copy.deepcopy(pretrainmodel.token_emb),
        scfoundation_token_emb2 = copy.deepcopy(pretrainmodel.token_emb),
        scfoundation_pos_emb1 = copy.deepcopy(pretrainmodel.pos_emb),
        scfoundation_pos_emb2 = copy.deepcopy(pretrainmodel.pos_emb),
        num_batch_labels=num_times, 
        do_mvc = False,
        use_batch_labels = False,
    )

    del pretrainmodel

    pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())
    post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

    logger.info(f"Total Pre freeze Params {(pre_freeze_param_count )}")
    logger.info(f"Total Post freeze Params {(post_freeze_param_count )}")

    model = nn.DataParallel(model, device_ids = [0,1,2,3,4,5,6,7])
    device = torch.device("cuda:0")
    model.to(device)

    ### Set the training parameters
    lr = 1e-4
    amp = True
    schedule_ratio = 0.99#0.9
    schedule_interval = 1
    log_interval = 10
    epochs = 100
    batch_size = 96

    criterion = masked_mse_loss
    criterion_dab = nn.CrossEntropyLoss()
    criterion_gep_gepc = masked_mse_loss
    optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, eps=1e-4 if amp else 1e-8
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, schedule_interval, gamma=schedule_ratio
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    ### Set the data preparation parameters
    pad_token = "<pad>"
    pad_value = 103
    mask_value = 102
    mask_ratio = 0.15

    tokenizer_dir = 'tokenizer/'
    vocab_file = tokenizer_dir + "scfoundation_gene_vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    vocab.append_token(pad_token)
    vocab.set_default_index(vocab[pad_token])

    ### Prepare the pretraining data
    dataset = 'GE_dataV6_aligned/'
    data_path = f'{dataset}/'
    slide_num = 0
    for slide in os.listdir(data_path):
        slide_num += 1
    logger.info(f"Train {epochs} epochs on dataset {dataset}: totally {slide_num} slides")

    ### Train the model on the dataset
    mse_train = []
    mse_valid = []
    for epoch in range(1, epochs + 1):
        slide_num = 0

        ###### random sslect 3 timepoint samples
        # Group samples by time point identifier (e.g., "P5" or "E115_lmge")
        groups = defaultdict(list)
        for full_name in time_points:
            time_point, sample_num = full_name.rsplit('_', 1)
            groups[time_point].append(full_name)

        # Extract numerical days and group identifiers by day
        day_to_identifiers = defaultdict(list)
        for identifier in groups.keys():
            if identifier.startswith('E'):
                day = int(identifier[1:].split('_')[0])  # Extract E-day (e.g., E115_lmge → 115)
            elif identifier.startswith('P'):
                day = int(identifier[1:].split('_')[0])  # Extract P-day (e.g., P5 → 5)
            day_to_identifiers[day].append(identifier)

        # Separate E and P days
        e_days = [day for day in day_to_identifiers.keys() if str(day).startswith('E') or isinstance(day, int) and day >= 100]
        p_days = [day for day in day_to_identifiers.keys() if str(day).startswith('P') or isinstance(day, int) and day < 100]

        # Sort E-days and P-days separately
        sorted_e_days = sorted(e_days)
        sorted_p_days = sorted(p_days)

        # Combine E-days first, then P-days
        sorted_days = sorted_e_days + sorted_p_days

        # Ensure there are at least 3 days to form a sequence
        if len(sorted_days) < 3:
            raise ValueError("Not enough days to select 3 adjacent time points.")
        num_triplets = len(sorted_days) - 2

        start_idx = (epoch - 1) % num_triplets
        selected_days = sorted_days[start_idx : start_idx + 3]
        
        # For each selected day, pick a random identifier and sample
        selected_samples = []
        for day in selected_days:
            identifiers = day_to_identifiers[day]
            chosen_identifier = random.choice(identifiers)
            samples = groups[chosen_identifier]
            selected_samples.append(random.choice(samples))

        # print("Selected samples:", selected_samples)
        print(f"Epoch {epoch}: Selected samples: {selected_samples}")

        for slide in ['all']:
            slide_num += 1
            logger.info(f"Training epoch {epoch} on dataset {dataset} - slide {slide_num} {slide}")
            slideData = SlideData(data_path, slide, vocab, mask_ratio, mask_value, pad_value, pad_token, selected_samples, save_hvg=False, load_hvg=False)
            slideData.get_niche_samples()
            slideData.tokenize_data()
            train_data_pt, valid_data_pt = slideData.prepare_data()
            train_and_evaluate(model, train_data_pt, valid_data_pt, epoch, batch_size, slide_num, slide, mse_train, mse_valid)
            pickle.dump(mse_train, open(f'model/mse_train_CLT.pkl', 'wb'))
            pickle.dump(mse_valid, open(f'model/mse_valid_CLT.pkl', 'wb'))
        scheduler.step()
        if epoch % 5 == 0:
            torch.save(model, f'model/model_CLT_{epoch}.ckpt')

    model.to('cpu')
    torch.save(model, f'model/model_CLT_final.ckpt')
    print('finial')
    
    
