import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from typing import Dict
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import scanpy as sc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from . import logger
from tokenizer import GeneVocab
from tokenizer import tokenize_and_pad_batch
from scipy.spatial import KDTree
from sklearn.preprocessing import LabelEncoder
import concurrent.futures
import pickle

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
    adata = sc.read_h5ad(path+f'{time_point}_xenium_updated.h5ad')
    adata.X = adata.X.astype(np.float32)
    ### spatial info
    coord = adata.obs[['coord_x','coord_y']].values
    coord = coord.astype(np.float32)
    adata.obsm['spatial'] = coord
    return adata

class SlideData():
    def __init__(self, data_path, slide, vocab, pad_value, pad_token, time_points=None, save_hvg=False, load_hvg=False):
        self.data_path = data_path
        self.slide = slide
        self.vocab = vocab
        self.pad_value = pad_value
        self.pad_token = pad_token
        self.save_hvg = save_hvg
        self.load_hvg = load_hvg
        # Define the full list of time_points (default list)
        full_time_points = [
                'P0_1', 
                'P5_1', 
                'P10_1', 
                'P14_1', 
                'P28_1',
                'P56_1',

                'E115_LMGE1',
                'E125_LMGE1',
                'E135_LMGE1',
                'E145_LMGE1',
                'E155_LMGE1', 
                'E185_LMGE1', 


                'E115_CGE1',
                'E125_CGE1',
                'E135_CGE1',
                'E145_CGE1',
                'E155_CGE1', 
                'E185_CGE1', 
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
        self.celltype_str = adata.obs['CellType']

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
        max_seq_len = np.max(np.count_nonzero(self.adata.X.A, axis=1))+2
        max_niche_cell_num = 3
        self.max_seq_len = max_seq_len
        self.max_niche_cell_num = max_niche_cell_num

        self.tokenized_data = tokenize_and_pad_batch(
            self.expression,
            self.ctprop,
            self.gene_ids,
            max_len = max_seq_len,
            max_niche_cell_num = max_niche_cell_num,
            vocab = self.vocab,
            pad_token = self.pad_token,
            pad_value = self.pad_value,
            append_cls = False,  # append <cls> token at the beginning
            include_zero_gene = False,
        )
     
        logger.info(
            f"Number of samples: {self.tokenized_data['center_genes'].shape[0]}, "
            f"\n\t feature length of center cell: {self.tokenized_data['center_genes'].shape[1]}"
            f"\n\t feature length of niche cells: {self.tokenized_data['niche_genes'].shape[1]}"
        )

    def prepare_data(self):
        self.data_pt = {
            "center_gene_ids": self.tokenized_data["center_genes"],
            "input_center_values": self.tokenized_data["center_values"],
            "target_center_values": self.tokenized_data["center_values"],
            "niche_gene_ids": self.tokenized_data["niche_genes"],
            "input_niche_values": self.tokenized_data["niche_values"],
            "niche_feature_lens": self.tokenized_data["niche_feature_lens"],
            "cross_attn_bias": self.tokenized_data["cross_attn_bias"],
            "celltype": self.celltype,
            "spatial": self.spatial,
            "batch": self.batch,
            "time": self.time,
        }

    def prepare_dataloader(self, batch_size):
        data_loader = DataLoader(
            dataset=SeqDataset(self.data_pt),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=len(os.sched_getaffinity(0)),
            pin_memory=True,
        )
        return data_loader
    
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["center_gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    cell_embeddings = []

    with torch.no_grad():
        for batch, batch_data in enumerate(tqdm(loader)):
            niche_feature_lens = batch_data["niche_feature_lens"].to(device)
            if niche_feature_lens.size(0)<batch_size:
                continue
            center_gene_ids = batch_data["center_gene_ids"].to(device)
            input_center_values = batch_data["input_center_values"].to(device)
            niche_gene_ids = batch_data["niche_gene_ids"].to(device)
            input_niche_values = batch_data["input_niche_values"].to(device)
            cross_attn_bias = batch_data["cross_attn_bias"].to(device)
            times = batch_data["time"].to(device)

            encoder_src_key_padding_mask = niche_gene_ids.eq(vocab[pad_token])
            decoder_src_key_padding_mask = center_gene_ids.eq(vocab[pad_token])

            with torch.cuda.amp.autocast(enabled=amp):
                output_dict = model(
                        niche_gene_ids,
                        input_niche_values,
                        encoder_src_key_padding_mask,
                        center_gene_ids,
                        input_center_values,
                        decoder_src_key_padding_mask,
                        cross_attn_bias,
                        times,
                    )
                cell_embeddings.append(output_dict['cell_emb'].to('cpu'))

    return(torch.cat(cell_embeddings))


pad_token = "<pad>"
pad_value = 103

tokenizer_dir = 'tokenizer/'
vocab_file = tokenizer_dir + "scfoundation_gene_vocab.json"
vocab = GeneVocab.from_file(vocab_file)
vocab.append_token(pad_token)
vocab.set_default_index(vocab[pad_token])

####
savepath = 'embeddings_result/'
os.makedirs(savepath, exist_ok=True)
slide = 'cge'

batch_size = 960
amp = True

time_points = [
            'P0_1', 
            'P5_1',
            'P10_1',
            'P14_1',
            'P28_1',
            'P56_1',

            # 'E115_LMGE1',
            # 'E125_LMGE1',
            # 'E135_LMGE1',
            # 'E145_LMGE1',
            # 'E155_LMGE1', 
            # 'E185_LMGE1', 

            'E115_CGE1',
            'E125_CGE1',
            'E135_CGE1',
            'E145_CGE1',
            'E155_CGE1', 
            'E185_CGE1', 
        ]


########### load model
num_times = np.unique([time_point.split('_')[0] for time_point in time_points]).shape[0]
# First load the state dict and remove the 'module' prefix
checkpoint = torch.load(f'model/model_CLT_final.ckpt', map_location='cpu')
if isinstance(checkpoint, nn.Module):
    state_dict = checkpoint.state_dict()
else:
    state_dict = checkpoint

# Remove the 'module.' prefix if it exists
new_state_dict = {}
for k, v in state_dict.items():
    name = k.replace("module.", "") if k.startswith("module.") else k
    new_state_dict[name] = v

# Create a new model instance
from model import TransformerModel
from tasks.scfoundation import load
import copy
pretrainmodel, pretrainconfig = load.load_model_frommmf('models/models.ckpt')
embsize = 768
d_hid = 3072
nhead = 12
nlayers = 6
dropout = 0.1
cell_emb_style = 'max-pool'

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

# Load the cleaned state dict
model.load_state_dict(new_state_dict)
# Set up DataParallel with available GPUs
available_gpus = [0,1,2,3,4,5,6,7]  # Specify your available GPU indices
device = torch.device("cuda:0")
model = nn.DataParallel(model, device_ids=available_gpus)
model.to(device)

#######
dataset = 'GE_dataV8/'
data_path = f'{dataset}/'

slideData = SlideData(data_path, slide, vocab, pad_value, pad_token, time_points=time_points, save_hvg=False, load_hvg=False)
slideData.get_niche_samples()
slideData.tokenize_data()
slideData.prepare_data()
data_loader = slideData.prepare_dataloader(batch_size)

cell_embeddings = evaluate(model, data_loader)

import anndata
adata = anndata.AnnData(obs=pd.DataFrame(index=range(cell_embeddings.size(0))), var=pd.DataFrame(index=slideData.adata.var['gene_name'].values), uns=None, obsm=None, varm=None, layers=None, raw=None)
adata.obsm['cell_embeddings'] = np.nan_to_num(cell_embeddings.cpu().numpy())
adata.obs['batch'] = slideData.batch_name[:cell_embeddings.size(0)]
adata.obs['celltype'] = slideData.celltype_str.tolist()[:cell_embeddings.size(0)]
adata.obsm['spatial'] = slideData.spatial[:cell_embeddings.size(0)]
adata.obs['cell_ids'] = slideData.adata.obs['cellid'].tolist()[:cell_embeddings.size(0)]
print(adata.obs['celltype'])
adata.X = np.concatenate([[s[0]] for s in slideData.expression[:cell_embeddings.size(0)]], axis=0)

# Option 1: Set scanpy figure directory before plotting
sc.settings.figdir = savepath

sc.pp.neighbors(adata, use_rep='cell_embeddings')
sc.tl.umap(adata)
sc.pl.umap(adata,
           title='',
           frameon=False,
           legend_loc='',
           legend_fontsize='xx-small',
           save=f"_CLT_{slide}_celltype.pdf",
           color='celltype')
sc.pl.umap(adata,
           title='',
           frameon=False,
           legend_loc='',
           legend_fontsize='xx-small',
           save=f"_CLT_{slide}_batch.pdf",
           color='batch')
print('finial')

### save
adata.write_h5ad(savepath+f'adata_CLT_{slide}.h5ad')
print('finial all')