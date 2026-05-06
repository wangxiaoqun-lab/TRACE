import sys
from pathlib import Path
import numpy as np
from my_paste import *
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

def split_adata(adata, n_per_part=7000):
    n_total = adata.n_obs
    n_parts = int(np.ceil(n_total / n_per_part))
    indices = np.random.permutation(n_total)
    split_indices = np.array_split(indices, n_parts)
    parts = [adata[idx].copy() for idx in split_indices]
    return parts

if __name__ == '__main__':
    # Get parameters from command-line arguments
    if len(sys.argv) < 4:
        print("Error: Please provide alpha, slice_name, and time_point_idx")
        print("Usage: python script.py <alpha> <slice_name> <time_point_idx>")
        sys.exit(1)
        
    alpha = float(sys.argv[1])
    slice_name = sys.argv[2]
    time_point_idx = int(sys.argv[3])

    # Define time points and verify index
    time_points = ['56-28', '28-14', '14-10', '10-5', '5-0', '0-185', '185-155', '155-145', '145-135', '135-125', '125-115', '0-155']
    if time_point_idx < 0 or time_point_idx >= len(time_points):
        print(f"Error: Invalid time_point_idx {time_point_idx}. Must be 0-{len(time_points)-1}")
        sys.exit(1)
    
    selected_time_point = time_points[time_point_idx]

    # Load data
    path_coords = f'embeddings_result/'
    save_path = 'output/'
    adata_all = sc.read_h5ad(path_coords+f'adata_CLT_{slice_name.lower()}.h5ad')

    adata_P0 = adata_all[adata_all.obs['batch']==f'P0_1']
    adata_P5 = adata_all[adata_all.obs['batch']==f'P5_1']
    adata_P10 = adata_all[adata_all.obs['batch']==f'P10_1']
    adata_P14 = adata_all[adata_all.obs['batch']==f'P14_1']
    adata_P28 = adata_all[adata_all.obs['batch']==f'P28_1']
    adata_P56 = adata_all[adata_all.obs['batch']==f'P56_1']

    if slice_name=='CGE':
        adata_E115 = adata_all[adata_all.obs['batch']==f'E115_{slice_name}1']
        adata_E125 = adata_all[adata_all.obs['batch']==f'E125_{slice_name}1']
        adata_E135 = adata_all[adata_all.obs['batch']==f'E135_{slice_name}1']
        adata_E145 = adata_all[adata_all.obs['batch']==f'E145_{slice_name}1']
        adata_E155 = adata_all[adata_all.obs['batch']==f'E155_{slice_name}1']
        adata_E185 = adata_all[adata_all.obs['batch']==f'E185_{slice_name}1']
    if slice_name=='LMGE':
        adata_E115 = adata_all[adata_all.obs['batch']==f'E115_{slice_name}1']
        adata_E125 = adata_all[adata_all.obs['batch']==f'E125_{slice_name}1']
        adata_E135 = adata_all[adata_all.obs['batch']==f'E135_{slice_name}1']
        adata_E145 = adata_all[adata_all.obs['batch']==f'E145_{slice_name}1']
        adata_E155 = adata_all[adata_all.obs['batch']==f'E155_{slice_name}1']
        adata_E185 = adata_all[adata_all.obs['batch']==f'E185_{slice_name}1']

    # Define stage pairs based on time_point_idx
    stage_pairs = [
        (adata_P56, adata_P28),  # 56-28
        (adata_P28, adata_P14),  # 28-14
        (adata_P14, adata_P10),  # 14-10
        (adata_P10, adata_P5),   # 10-5
        (adata_P5, adata_P0),     # 5-0
        (adata_P0, adata_E185), 
        (adata_E185, adata_E155),
        (adata_E155, adata_E145),
        (adata_E145, adata_E135),
        (adata_E135, adata_E125),
        (adata_E125, adata_E115), 
        (adata_P0, adata_E155), # new add
    ]
    
    # Get the specific stage pair for this time point
    stage1_adata, stage2_adata = stage_pairs[time_point_idx]

    x_coords = 5500
    stage1_adata = stage1_adata[stage1_adata.obsm['spatial'][:,0] < x_coords]
    stage2_adata = stage2_adata[stage2_adata.obsm['spatial'][:,0] < x_coords]

    ### control the order by nums 260109
    if stage1_adata.shape[0]<stage2_adata.shape[0]:
        temp=stage2_adata
        stage2_adata=stage1_adata
        stage1_adata=temp
        selected_time_point = "-".join(selected_time_point.split("-")[::-1])

    max_nums = 50000  # Maximum cells per partition
    map_df_list = []  # To collect results from all partitions

    # Split if stage1 has too many cells
    if stage1_adata.n_obs > max_nums:
        parts = split_adata(stage1_adata, n_per_part=max_nums)
    else:
        parts = [stage1_adata]

    # Process each partition
    for part_idx, stage1_part in enumerate(parts):
        # Run cell directions for this partition
        map_result = cell_directions(
            adataA=stage1_part, 
            adataB=stage2_adata,
            numItermaxEmd=500000,
            layer="cell_embeddings",
            spatial_key="spatial",
            key_added="cells_mapping",
            alpha=alpha,
            device='cpu',#'0',
            inplace=True
        )

        # Extract mapping results
        index1 = map_result[0]['index_x'].values
        index2 = map_result[0]['index_y'].values
        pi_value = map_result[0]['pi_value'].values
        
        # Get cell IDs using original indices
        stage1_cells = stage1_part.obs['cell_ids'].reset_index(drop=True)
        stage2_cells = stage2_adata.obs['cell_ids'].reset_index(drop=True)
        
        # Create mapping DataFrame
        map_df = pd.DataFrame({
            'index1': stage1_cells.iloc[index1].values,
            'index2': stage2_cells.iloc[index2].values,
            'pi_value': pi_value
        })
        
        map_df_list.append(map_df)

    # Combine all partition results
    if map_df_list:
        map_df_con = pd.concat(map_df_list, ignore_index=True)
        
        # Create output directory if needed
        output_dir = Path(save_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with parameter info in filename
        output_path = output_dir / f'LLM_mapping_P{selected_time_point}_slice{slice_name}_alpha{alpha}.csv'
        map_df_con.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")
    else:
        print(f"No mappings generated for {selected_time_point}, slice {slice_name}, alpha {alpha}")

    print(f'Completed: {selected_time_point}, alpha={alpha}, slice={slice_name}')