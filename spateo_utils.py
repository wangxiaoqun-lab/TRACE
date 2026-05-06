### This file is copy from spateo
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import ot
import pandas as pd
import torch
from anndata import AnnData
from numpy import ndarray
from scipy.linalg import pinv
from scipy.sparse import issparse
from scipy.special import psi

# Get the intersection of lists
intersect_lsts = lambda *lsts: list(set(lsts[0]).intersection(*lsts[1:]))

# Covert a sparse matrix into a dense np array
to_dense_matrix = lambda X: X.toarray() if issparse(X) else np.array(X)

# Returns the data matrix or representation
extract_data_matrix = lambda adata, rep: adata.X if rep is None else adata.layers[rep]


#########################
# Check data and device #
#########################

# Finished
def check_backend(device: str = "cpu", dtype: str = "float32", verbose: bool = True):
    """
    Check the proper backend for the device.

    Args:
        device: Equipment used to run the program. You can also set the specified GPU for running. E.g.: '0'.
        dtype: The floating-point number type. Only float32 and float64.
        verbose: If ``True``, print progress updates.

    Returns:
        backend: The proper backend.
        type_as: The type_as.device is the device used to run the program and the type_as.dtype is the floating-point number type.
    """
    if device == "cpu":
        backend = ot.backend.NumpyBackend()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        if torch.cuda.is_available():
            torch.cuda.init()
            backend = ot.backend.NumpyBackend()
            backend = ot.backend.NumpyBackend()
            backend = ot.backend.TorchBackend()
        else:
            backend = ot.backend.NumpyBackend()

    if nx_torch(backend):
        type_as = backend.__type_list__[-2] if dtype == "float32" else backend.__type_list__[-1]
    else:
        type_as = backend.__type_list__[0] if dtype == "float32" else backend.__type_list__[1]
    return backend, type_as


# Finished
def check_spatial_coords(sample: AnnData, spatial_key: str = "spatial") -> np.ndarray:
    """
    Check and return the spatial coordinate information from an AnnData object.

    Args:
        sample (AnnData): An AnnData object containing the sample data.
        spatial_key (str, optional): The key in `.obsm` that corresponds to the raw spatial coordinates. Defaults to "spatial".

    Returns:
        np.ndarray: The spatial coordinates.

    Raises:
        KeyError: If the specified spatial_key is not found in `sample.obsm`.
    """

    if spatial_key not in sample.obsm:
        raise KeyError(f"Spatial key '{spatial_key}' not found in AnnData object.")

    coordinates = sample.obsm[spatial_key].copy()
    if isinstance(coordinates, pd.DataFrame):
        coordinates = coordinates.values

    mask = []
    for i in range(coordinates.shape[1]):
        mask.append(i)

    # Select only dimensions with more than one unique value
    coordinates = coordinates[:, mask]

    if coordinates.shape[1] > 3 or coordinates.shape[1] < 2:
        raise ValueError(f"The spatial coordinate '{spatial_key}' should only has 2 / 3 dimension")

    return np.asarray(coordinates)


# Finished
def check_exp(sample: AnnData, layer: str = "X") -> np.ndarray:
    """
    Check expression matrix.

    Args:
        sample (AnnData): An AnnData object containing the sample data.
        layer (str, optional): The key in `.layers` that corresponds to the expression matrix. Defaults to "X".

    Returns:
        The expression matrix.

    Raises:
        KeyError: If the specified layer is not found in `sample.layers`.
    """

    if layer == "X":
        exp_matrix = sample.X.copy()
    else:
        if layer not in sample.layers:
            raise KeyError(f"Layer '{layer}' not found in AnnData object.")
        exp_matrix = sample.layers[layer].copy()

    exp_matrix = to_dense_matrix(exp_matrix)
    return exp_matrix


# Finished
def check_obs(rep_layer: List[str], rep_field: List[str]) -> Optional[str]:
    """
    Check that the number of occurrences of 'obs' in the list of representation fields is no more than one.

    Args:
        rep_layer (List[str]): A list of representations to check.
        rep_field (List[str]): A list of representation types corresponding to the representations in `rep_layer`.

    Returns:
        Optional[str]: The representation key if 'obs' occurs exactly once, otherwise None.

    Raises:
        ValueError: If 'obs' occurs more than once in the list.
    """

    count = 0
    position = -1

    for i, s in enumerate(rep_field):
        if s == "obs":
            count += 1
            position = i
            if count > 1:
                raise ValueError(
                    f"'obs' occurs more than once in the list. Currently Spateo only support one label consistency."
                )

    # Return the 'obs' key if found exactly once
    if count == 1:
        return rep_layer[position]
    else:
        return None


# Finished
def check_rep_layer(
    samples: List[AnnData],
    rep_layer: Union[str, List[str]] = "X",
    rep_field: Union[str, List[str]] = "layer",
) -> bool:
    """
    Check if specified representations exist in the `.layers`, `.obsm`, or `.obs` attributes of AnnData objects.

    Args:
        samples (List[AnnData]):
            A list of AnnData objects containing the data samples.
        rep_layer (Union[str, List[str]], optional):
            The representation layer(s) to check. Defaults to "X".
        rep_field (Union[str, List[str]], optional):
            The field(s) indicating the type of representation. Acceptable values are "layer", "obsm", and "obs". Defaults to "layer".

    Returns:
        bool:
            True if all specified representations exist in the corresponding attributes of all AnnData objects, False otherwise.

    Raises:
        ValueError:
            If the specified representation is not found in the specified attribute or if the attribute type is invalid.
    """

    for sample in samples:
        for rep, rep_f in zip(rep_layer, rep_field):
            if rep_f == "layer":
                if (rep != "X") and (rep not in sample.layers):
                    raise ValueError(
                        f"The specified representation '{rep}' not found in the '{rep_f}' attribute of some of the AnnData objects."
                    )
            elif rep_f == "obsm":
                if rep not in sample.obsm:
                    raise ValueError(
                        f"The specified representation '{rep}' not found in the '{rep_f}' attribute of some of the AnnData objects."
                    )
            elif rep_f == "obs":
                if rep not in sample.obs:
                    raise ValueError(
                        f"The specified representation '{rep}' not found in the '{rep_f}' attribute of some of the AnnData objects."
                    )

                # judge if the sample.obs[rep] is categorical
                if not isinstance(sample.obs[rep].dtype, pd.CategoricalDtype):
                    raise ValueError(
                        f"The specified representation '{rep}' found in the '{rep_f}' attribute should be categorical."
                    )
            else:
                raise ValueError("rep_field must be either 'layer', 'obsm' or 'obs'")
    return True


# Finished
def check_label_transfer_dict(
    catA: List[str],
    catB: List[str],
    label_transfer_dict: Dict[str, Dict[str, float]],
):
    """
    Check the label transfer dictionary for consistency with given categories.

    Args:
        catA (List[str]):
            List of category labels from the first dataset.
        catB (List[str]):
            List of category labels from the second dataset.
        label_transfer_dict (Dict[str, Dict[str, float]]):
            Dictionary defining the transfer probabilities between categories.

    Raises:
        KeyError:
            If a category from `catA` is not found in `label_transfer_dict`.
        KeyError:
            If a category from `catB` is not found in the nested dictionary of `label_transfer_dict`.
    """

    for ca in catA:
        if ca in label_transfer_dict.keys():
            for cb in catB:
                if cb not in label_transfer_dict[ca].keys():
                    raise KeyError(
                        f"Category '{cb}' from catB not found in label_transfer_dict for category '{ca}' from catA."
                    )

        else:
            raise KeyError(f"Category '{ca}' from catA not found in label_transfer_dict.")


# Finished
def check_label_transfer(
    nx: Union[ot.backend.TorchBackend, ot.backend.NumpyBackend],
    type_as: Union[torch.Tensor, np.ndarray],
    samples: List[AnnData],
    obs_key: str,
    label_transfer_dict: Optional[List[Dict[str, Dict[str, float]]]] = None,
) -> List[Union[np.ndarray, torch.Tensor]]:
    """
    Check and generate label transfer matrices for the given samples.

    Args:
        nx (module):
            Backend module (e.g., numpy or torch).
        type_as (type):
            Type to which the output should be cast.
        samples (List[AnnData]):
            List of AnnData objects containing the samples.
        obs_key (str):
            The key in `.obs` that corresponds to the labels.
        label_transfer_dict (Optional[List[Dict[str, Dict[str, float]]]], optional):
            List of dictionaries defining the label transfer cost between categories of each pair of samples. Defaults to None.

    Returns:
        List[Union[np.ndarray, torch.Tensor]]:
            List of label transfer matrices, each as either a NumPy array or torch Tensor.

    Raises:
        ValueError:
            If the length of `label_transfer_dict` does not match `len(samples) - 1`.
    """

    if label_transfer_dict is not None:
        if isinstance(label_transfer_dict, dict):
            label_transfer_dict = [label_transfer_dict]
        if isinstance(label_transfer_dict, list):
            if len(label_transfer_dict) != (len(samples) - 1):
                raise ValueError("The length of label_transfer_dict must be equal to len(samples) - 1.")
        else:
            raise ValueError("label_transfer_dict should be a list or a dictionary.")

    label_transfer = []
    for i in range(len(samples) - 1):
        catB = samples[i].obs[obs_key].cat.categories.tolist()
        catA = samples[i + 1].obs[obs_key].cat.categories.tolist()
        cur_label_transfer = np.zeros(len(catA), len(catB), dtype=np.float32)

        if label_transfer_dict is not None:
            cur_label_transfer_dict = label_transfer_dict[i]
            check_label_transfer_dict(catA=catA, catB=catB, label_transfer_dict=cur_label_transfer_dict)
        else:
            cur_label_transfer_dict = generate_label_transfer_dict(catA, catB)

        for j, ca in enumerate(catA):
            for k, cb in enumerate(catB):
                cur_label_transfer[j, k] = cur_label_transfer_dict[ca][cb]
        label_transfer.append(nx.from_numpy(cur_label_transfer, type_as=type_as))

    return label_transfer


# Finished
def generate_label_transfer_dict(
    cat1: List[str],
    cat2: List[str],
    positive_pairs: Optional[List[Dict[str, Union[List[str], float]]]] = None,
    negative_pairs: Optional[List[Dict[str, Union[List[str], float]]]] = None,
    default_positve_value: float = 10.0,
) -> Dict[str, Dict[str, float]]:
    """
    Generate a label transfer dictionary with normalized values.

    Args:
        cat1 (List[str]):
            List of categories from the first dataset.
        cat2 (List[str]):
            List of categories from the second dataset.
        positive_pairs (Optional[List[Dict[str, Union[List[str], float]]]], optional):
            List of positive pairs with transfer values. Each dictionary should have 'left', 'right', and 'value' keys. Defaults to None.
        negative_pairs (Optional[List[Dict[str, Union[List[str], float]]]], optional):
            List of negative pairs with transfer values. Each dictionary should have 'left', 'right', and 'value' keys. Defaults to None.
        default_positive_value (float, optional):
            Default value for positive pairs if none are provided. Defaults to 10.0.

    Returns:
        Dict[str, Dict[str, float]]:
            A normalized label transfer dictionary.
    """

    # Initialize label transfer dictionary with default values
    label_transfer_dict = {c2: {c1: 1.0 for c1 in cat1} for c2 in cat2}

    # Generate default positive pairs if none provided
    if (positive_pairs is None) and (negative_pairs is None):
        common_cat = np.union1d(cat1, cat2)
        positive_pairs = [{"left": [c], "right": [c], "value": default_positve_value} for c in common_cat]

    # Apply positive pairs to the dictionary
    if positive_pairs is not None:
        for p in positive_pairs:
            for l in p["left"]:
                for r in p["right"]:
                    if r in label_transfer_dict and l in label_transfer_dict[r]:
                        label_transfer_dict[r][l] = p["value"]

    # Apply negative pairs to the dictionary
    if negative_pairs is not None:
        for p in negative_pairs:
            for l in p["left"]:
                for r in p["right"]:
                    if r in label_transfer_dict and l in label_transfer_dict[r]:
                        label_transfer_dict[r][l] = p["value"]

    # Normalize the label transfer dictionary
    norm_label_transfer_dict = dict()
    for c2 in cat2:
        norm_c = np.array([label_transfer_dict[c2][c1] for c1 in cat1]).sum()
        norm_label_transfer_dict[c2] = {c1: label_transfer_dict[c2][c1] / norm_c for c1 in cat1}

    return norm_label_transfer_dict


# Finished
def get_rep(
    nx: Union[ot.backend.TorchBackend, ot.backend.NumpyBackend],
    type_as: Union[torch.Tensor, np.ndarray],
    sample: AnnData,
    rep: str = "X",
    rep_field: str = "layer",
    genes: Optional[Union[list, np.ndarray]] = None,
) -> np.ndarray:
    """
    Get the specified representation from the AnnData object.

    Args:
        nx (module): Backend module (e.g., numpy or torch).
        type_as (type): Type to which the output should be cast.
        sample (AnnData): The AnnData object containing the sample data.
        rep (str, optional): The name of the representation to retrieve. Defaults to "X".
        rep_field (str, optional): The type of representation. Acceptable values are "layer", "obs" and "obsm". Defaults to "layer".
        genes (Optional[Union[list, np.ndarray]], optional): List of genes to filter if `rep_field` is "layer". Defaults to None.

    Returns:
        Union[np.ndarray, torch.Tensor]: The requested representation from the AnnData object, cast to the specified type.

    Raises:
        ValueError: If `rep_field` is not one of the expected values.
        KeyError: If the specified representation is not found in the AnnData object.
    """

    # gene expression stored in ".layer" field
    if rep_field == "layer":
        representation = nx.from_numpy(check_exp(sample=sample[:, genes], layer=rep), type_as=type_as)

    # label information stored in ".obs" field
    elif rep_field == "obs":
        # Sort categories and convert to integer codes
        representation = sample.obs[rep].cat.codes.values
        representation = nx.from_numpy(representation)
        if nx_torch(nx):
            representation = representation.to(type_as.device)

    # scalar values stored in ".obsm" field
    elif rep_field == "obsm":
        representation = nx.from_numpy(sample.obsm[rep], type_as=type_as)
    else:
        raise ValueError("rep_field must be either 'layer', 'obsm' or 'obs'")

    return representation


######################
# Data preprocessing #
######################

# Finished
def filter_common_genes(*genes, verbose: bool = True) -> list:
    """
    Filters for the intersection of genes between all samples.

    Args:
        genes: List of genes.
        verbose: If ``True``, print progress updates.
    """

    common_genes = intersect_lsts(*genes)
    if len(common_genes) == 0:
        raise ValueError("The number of common gene between all samples is 0.")
    else:
        return common_genes


# Finished
def normalize_coords(
    nx: Union[ot.backend.TorchBackend, ot.backend.NumpyBackend],
    coords: List[Union[np.ndarray, torch.Tensor]],
    verbose: bool = True,
    separate_scale: bool = True,
    separate_mean: bool = True,
) -> Tuple[
    List[Union[np.ndarray, torch.Tensor]], List[Union[np.ndarray, torch.Tensor]], List[Union[np.ndarray, torch.Tensor]]
]:
    """
    Normalize the spatial coordinate.

    Parameters
    ----------
    coords : List[Union[np.ndarray, torch.Tensor]]
        Spatial coordinates of the samples. Each element in the list can be a numpy array or a torch tensor.
    nx : Union[ot.backend.TorchBackend, ot.backend.NumpyBackend], optional
        The backend to use for computations. Default is `ot.backend.NumpyBackend`.
    verbose : bool, optional
        If `True`, print progress updates. Default is `True`.
    separate_scale : bool, optional
        If `True`, normalize each coordinate axis independently. When doing the global refinement, this weill be set to False. Default is `True`.
    separate_mean : bool, optional
        If `True`, normalize each coordinate axis to have zero mean independently. When doing the global refinement, this weill be set to False. Default is `True`.

    Returns
    -------
    Tuple[List[Union[np.ndarray, torch.Tensor]], List[Union[np.ndarray, torch.Tensor]], List[Union[np.ndarray, torch.Tensor]]]
        A tuple containing:
        - coords: List of normalized spatial coordinates.
        - normalize_scales: List of normalization scale factors applied to each coordinate axis.
        - normalize_means: List of mean values used for normalization of each coordinate axis.
    """

    D = coords[0].shape[1]
    normalize_scales = nx.zeros((len(coords),), type_as=coords[0])
    normalize_means = nx.zeros((len(coords), D), type_as=coords[0])

    # get the means for each coords
    for i in range(len(coords)):
        normalize_mean = nx.einsum("ij->j", coords[i]) / coords[i].shape[0]
        normalize_means[i] = normalize_mean

    # get the global means for whole coords if "separate_mean" is True
    if not separate_mean:
        global_mean = nx.mean(normalize_means, axis=0)
        normalize_means = nx.full((len(coords), D), global_mean)

    # move each coords to zero center and calculate the normalization scale
    for i in range(len(coords)):
        coords[i] -= normalize_means[i]
        normalize_scale = nx.sqrt(nx.einsum("ij->", nx.einsum("ij,ij->ij", coords[i], coords[i])) / coords[i].shape[0])
        normalize_scales[i] = normalize_scale

    # get the global scale for whole coords if "separate_scale" is True
    if not separate_scale:
        global_scale = nx.mean(normalize_scales)
        normalize_scales = nx.full((len(coords),), global_scale)

    # normalize the scale of the coords
    for i in range(len(coords)):
        coords[i] /= normalize_scales[i]

    return coords, normalize_scales, normalize_means


# Finished
def normalize_exps(
    nx: Union[ot.backend.TorchBackend, ot.backend.NumpyBackend],
    exp_layers: List[List[Union[np.ndarray, torch.Tensor]]],
    rep_field: Union[str, List[str]] = "layer",
    verbose: bool = True,
) -> List[List[Union[np.ndarray, torch.Tensor]]]:
    """
    Normalize the gene expression matrices.

    Args:
        nx (Union[ot.backend.TorchBackend, ot.backend.NumpyBackend], optional):
            The backend to use for computations. Defaults to `ot.backend.NumpyBackend`.
        exp_layers (List[List[Union[np.ndarray, torch.Tensor]]]):
            Gene expression and optionally the representation matrices of the samples.
            Each element in the list can be a numpy array or a torch tensor.
        rep_field (Union[str, List[str]], optional):
            Field(s) indicating the type of representation. If 'layer', normalization can be applied.
            Defaults to "layer".
        verbose (bool, optional):
            If `True`, print progress updates. Default is `True`.

    Returns:
        List[List[Union[np.ndarray, torch.Tensor]]]:
            A list of lists containing normalized gene expression matrices.
            Each matrix in the list is a numpy array or a torch tensor.
    """

    if isinstance(rep_field, str):
        rep_field = [rep_field] * len(exp_layers[0])

    for i, rep_f in enumerate(rep_field):
        if rep_f == "layer":
            normalize_scale = 0

            # Calculate the normalization scale
            for l in range(len(exp_layers)):
                normalize_scale += nx.sqrt(
                    nx.einsum("ij->", nx.einsum("ij,ij->ij", exp_layers[i][l], exp_layers[i][l]))
                    / exp_layers[i][l].shape[0]
                )

            normalize_scale /= len(exp_layers)

            # Apply the normalization scale
            for i in range(len(exp_layers)):
                exp_layers[i][l] /= normalize_scale

    return exp_layers


# Finished
def align_preprocess(
    samples: List[AnnData],
    rep_layer: Union[str, List[str]] = "X",
    rep_field: Union[str, List[str]] = "layer",
    genes: Optional[Union[list, np.ndarray]] = None,
    spatial_key: str = "spatial",
    label_transfer_dict: Optional[Union[dict, List[dict]]] = None,
    normalize_c: bool = False,
    normalize_g: bool = False,
    dtype: str = "float64",
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[
    Union[ot.backend.TorchBackend, ot.backend.NumpyBackend],
    Union[torch.Tensor, np.ndarray],
    List[List[Union[np.ndarray, torch.Tensor]]],
    List[Union[np.ndarray, torch.Tensor]],
    Union[torch.Tensor, np.ndarray],
    Union[torch.Tensor, np.ndarray],
    Union[torch.Tensor, np.ndarray],
]:
    """
    Preprocess the data before alignment.

    Parameters
    ----------
    samples : List[AnnData]
        A list of AnnData objects containing the data samples.
    genes : Optional[Union[list, np.ndarray]], optional
        Genes used for calculation. If None, use all common genes for calculation. Default is None.
    spatial_key : str, optional
        The key in `.obsm` that corresponds to the raw spatial coordinates. Default is "spatial".
    layer : str, optional
        If 'X', uses `sample.X` to calculate dissimilarity between spots, otherwise uses the representation given by `sample.layers[layer]`. Default is "X".
    use_rep : Optional[Union[str, List[str]]], optional
        Specify the representation to use. If None, do not use the representation.
    rep_type : Optional[Union[str, List[str]]], optional
        Specify the type of representation. Accept types: "obs" and "obsm". If None, use the "obsm" type.
    normalize_c : bool, optional
        Whether to normalize spatial coordinates. Default is False.
    normalize_g : bool, optional
        Whether to normalize gene expression. Default is False.
    dtype : str, optional
        The floating-point number type. Only float32 and float64 are allowed. Default is "float64".
    device : str, optional
        The device used to run the program. Can specify the GPU to use, e.g., '0'. Default is "cpu".
    verbose : bool, optional
        If True, print progress updates. Default is True.

    Returns
    -------
    Tuple
        A tuple containing the following elements:
        - backend: The backend used for computations (TorchBackend or NumpyBackend).
        - type_as: The type used for computations which contains the dtype and device.
        - exp_layers: A list of processed expression layers.
        - spatial_coords: A list of spatial coordinates.
        - normalize_scales: Optional scaling factors for normalization.
        - normalize_means: Optional mean values for normalization.

    Raises
    ------
    ValueError
        If the specified representation is not found in the attributes of the AnnData objects.
    AssertionError
        If the spatial coordinate dimensions are different.
    """

    # Determine if gpu or cpu is being used
    nx, type_as = check_backend(device=device, dtype=dtype)

    # Check if the representation is in the AnnData objects
    if rep_layer is not None:
        if rep_field is None:
            rep_field = "layer"

        if isinstance(rep_layer, str):
            rep_layer = [rep_layer]

        if isinstance(rep_field, str):
            rep_field = [rep_field] * len(rep_layer)

        # if not check_use_rep(samples, use_rep, rep_type):
        if not check_rep_layer(samples=samples, rep_layer=rep_layer, rep_field=rep_field):
            raise ValueError(f"The specified representation is not found in the attribute of the AnnData objects.")

        obs_key = check_obs(rep_layer, rep_field)
    else:
        raise ValueError(
            "No representation input is detected, which may not produce meaningful result. Please check the rep_layer and rep_field."
        )

    # Get the common genes
    all_samples_genes = [s[0].var.index for s in samples]
    common_genes = filter_common_genes(*all_samples_genes, verbose=verbose)
    common_genes = common_genes if genes is None else intersect_lsts(common_genes, genes)

    # Extract the gene expression / representations of all samples, where each representation has a layer
    exp_layers = []
    for s in samples:
        cur_layer = []
        if rep_layer is not None:
            for rep, rep_f in zip(rep_layer, rep_field):
                cur_layer.append(
                    get_rep(nx=nx, type_as=type_as, sample=s, rep=rep, rep_field=rep_f, genes=common_genes)
                )
        exp_layers.append(cur_layer)

    # check the label tranfer dictionary and generate a matrix that contains the label transfer cost and cast to the specified type
    if obs_key is not None:
        label_transfer = check_label_transfer(nx, type_as, samples, obs_key, label_transfer_dict)
    else:
        label_transfer = None

    # Spatial coordinates of all samples
    spatial_coords = [
        nx.from_numpy(check_spatial_coords(sample=s, spatial_key=spatial_key), type_as=type_as) for s in samples
    ]

    # check the spatial coordinates dimensionality
    coords_dims = nx.unique(_data(nx, [c.shape[1] for c in spatial_coords], type_as))
    assert len(coords_dims) == 1, "Spatial coordinate dimensions are different, please check again."

    # Normalize spatial coordinates if required
    if normalize_c:
        spatial_coords, normalize_scales, normalize_means = normalize_coords(
            coords=spatial_coords, nx=nx, verbose=verbose
        )
    else:
        normalize_scales, normalize_means = None, None

    # Normalize gene expression if required
    if normalize_g:
        exp_layers = normalize_exps(nx=nx, matrices=exp_layers, rep_field=rep_field, verbose=verbose)

    return (
        nx,
        type_as,
        exp_layers,
        spatial_coords,
        label_transfer,
        normalize_scales,
        normalize_means,
        common_genes,
    )


# Finished
def guidance_pair_preprocess(
    nx: Union[ot.backend.TorchBackend, ot.backend.NumpyBackend],
    type_as: Union[torch.Tensor, np.ndarray],
    guidance_pair: List[np.ndarray],
    normalize_scales: Union[torch.Tensor, np.ndarray],
    normalize_means: Union[torch.Tensor, np.ndarray],
) -> List[Union[torch.Tensor, np.ndarray]]:
    """
    Preprocess guidance pairs by normalizing them.

    Args:
        nx (Union[ot.backend.TorchBackend, ot.backend.NumpyBackend], optional):
            Backend module for computations (e.g., numpy or torch). Defaults to `ot.backend.NumpyBackend`.
        type_as (Union[torch.Tensor, np.ndarray]):
            Type to which the output should be cast.
        guidance_pair (List[np.ndarray]):
            List containing the guidance pairs as numpy arrays.
        normalize_scales (Union[torch.Tensor, np.ndarray]):
            Tensor or array of normalization scales.
        normalize_means (Union[torch.Tensor, np.ndarray]):
            Tensor or array of normalization means.

    Returns:
        List[Union[torch.Tensor, np.ndarray]]:
            List containing the normalized guidance pairs.

    """

    # Convert guidance pairs to the backend type
    X_BI = nx.from_numpy(guidance_pair[0], type_as=type_as)
    X_AI = nx.from_numpy(guidance_pair[1], type_as=type_as)

    # Extract normalization parameters
    normalize_scale = normalize_scales[0]
    normalize_mean_ref = normalize_means[0]
    normalize_mean_query = normalize_means[1]

    # Normalize the guidance pairs
    X_AI = (X_AI - normalize_mean_query) / normalize_scale
    X_BI = (X_BI - normalize_mean_ref) / normalize_scale
    return [X_AI, X_BI]

##############################################
# Calculate  dissimilarity / distance matrix #
##############################################


def _kl_distance_backend(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    probabilistic: bool = True,
    eps: float = 1e-8,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute the pairwise KL divergence between all pairs of samples in matrices X and Y.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Matrix with shape (N, D), where each row represents a sample.
    Y : np.ndarray or torch.Tensor
        Matrix with shape (M, D), where each row represents a sample.
    probabilistic : bool, optional
        If True, normalize the rows of X and Y to sum to 1 (to interpret them as probabilities).
        Default is True.
    eps : float, optional
        A small value to avoid division by zero. Default is 1e-8.

    Returns
    -------
    np.ndarray
        Pairwise KL divergence matrix with shape (N, M).

    Raises
    ------
    AssertionError
        If the number of features in X and Y do not match.
    """

    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    # Get the appropriate backend (either NumPy or PyTorch)
    nx = ot.backend.get_backend(X, Y)

    # Normalize rows to sum to 1 if probabilistic is True
    if probabilistic:
        X = X / nx.sum(X, axis=1, keepdims=True)
        Y = Y / nx.sum(Y, axis=1, keepdims=True)

    # Compute log of X and Y
    log_X = nx.log(X + 1e-8)  # Adding epsilon to avoid log(0)
    log_Y = nx.log(Y + 1e-8)  # Adding epsilon to avoid log(0)

    # Compute X log X and the pairwise KL divergence
    X_log_X = nx.sum(X * log_X, axis=1, keepdims=True)
    D = X_log_X - nx.dot(X, log_Y.T)

    return D


def _cosine_distance_backend(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-8,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute the pairwise cosine similarity between all pairs of samples in matrices X and Y.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Matrix with shape (N, D), where each row represents a sample.
    Y : np.ndarray or torch.Tensor
        Matrix with shape (M, D), where each row represents a sample.
    eps : float, optional
        A small value to avoid division by zero. Default is 1e-8.

    Returns
    -------
    np.ndarray or torch.Tensor
        Pairwise cosine similarity matrix with shape (N, M).

    Raises
    ------
    AssertionError
        If the number of features in X and Y do not match.
    """

    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    # Get the appropriate backend (either NumPy or PyTorch)
    nx = ot.backend.get_backend(X, Y)

    # Normalize rows to unit vectors
    X_norm = nx.sqrt(nx.sum(X**2, axis=1, keepdims=True))
    Y_norm = nx.sqrt(nx.sum(Y**2, axis=1, keepdims=True))
    X = X / nx.maximum(X_norm, eps)
    Y = Y / nx.maximum(Y_norm, eps)

    # Compute cosine similarity
    D = nx.dot(X, Y.T)

    return D


def _euc_distance_backend(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    squared: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute the pairwise Euclidean distance between all pairs of samples in matrices X and Y.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Matrix with shape (N, D), where each row represents a sample.
    Y : np.ndarray or torch.Tensor
        Matrix with shape (M, D), where each row represents a sample.
    squared : bool, optional
        If True, return squared Euclidean distances. Default is True.

    Returns
    -------
    np.ndarray or torch.Tensor
        Pairwise Euclidean distance matrix with shape (N, M).

    Raises
    ------
    AssertionError
        If the number of features in X and Y do not match.
    """

    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    # Get the appropriate backend (either NumPy or PyTorch)
    nx = ot.backend.get_backend(X, Y)

    D = nx.sum(X**2, 1)[:, None] + nx.sum(Y**2, 1)[None, :] - 2 * nx.dot(X, Y.T)

    # Ensure non-negative distances (can arise due to floating point arithmetic)
    D = nx.maximum(D, 0.0)

    if not squared:
        D = nx.sqrt(D)

    return D


def _label_distance_backend(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    label_transfer: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Generate a matrix of size (N, M) by indexing into the label_transfer matrix using the values in X and Y.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Array with shape (N, ) containing integer values ranging from 0 to K.
    Y : np.ndarray or torch.Tensor
        Array with shape (M, ) containing integer values ranging from 0 to L.
    label_transfer : np.ndarray or torch.Tensor
        Matrix with shape (K, L) containing the label transfer cost.

    Returns
    -------
    np.ndarray or torch.Tensor
        Matrix with shape (N, M) where each element is the value from label_transfer indexed by the corresponding values in X and Y.

    Raises
    ------
    AssertionError
        If the shape of X or Y is not one-dimensional or if they contain non-integer values.
    """
    assert X.ndim == 1, "X should be a 1-dimensional array."
    assert Y.ndim == 1, "Y should be a 1-dimensional array."

    nx = ot.backend.get_backend(X, Y, label_transfer)

    if nx_torch(nx):
        assert not (torch.is_floating_point(X) or torch.is_floating_point(Y)), "X and Y should contain integer values."
    else:
        assert np.issubdtype(X.dtype, np.integer) and np.issubdtype(
            X.dtype, np.integer
        ), "X should contain integer values."

    D = label_transfer[X, :][:, Y]

    return D


# TODO: finish these


def _correlation_distance_backend(X, Y):
    pass


def _jaccard_distance_backend(X, Y):
    pass


def _chebyshev_distance_backend(X, Y):
    pass


def _canberra_distance_backend(X, Y):
    pass


def _braycurtis_distance_backend(X, Y):
    pass


def _hamming_distance_backend(X, Y):
    pass


def _minkowski_distance_backend(X, Y):
    pass


def calc_distance(
    X: Union[List[Union[np.ndarray, torch.Tensor]], Union[np.ndarray, torch.Tensor]],
    Y: Union[List[Union[np.ndarray, torch.Tensor]], Union[np.ndarray, torch.Tensor]],
    metric: Union[List[str], str] = "euc",
    label_transfer: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate the distance between all pairs of samples in matrices X and Y using the specified metric.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Matrix with shape (N, D), where each row represents a sample.
    Y : np.ndarray or torch.Tensor
        Matrix with shape (M, D), where each row represents a sample.
    metric : str, optional
        The metric to use for calculating distances. Options are 'euc', 'euclidean', 'square_euc', 'square_euclidean',
        'kl', 'sym_kl', 'cos', 'cosine', 'label'. Default is 'euc'.
    label_transfer : Optional[np.ndarray or torch.Tensor], optional
        Matrix with shape (K, L) containing the label transfer cost. Required if metric is 'label'. Default is None.

    Returns
    -------
    np.ndarray or torch.Tensor
        Pairwise distance matrix with shape (N, M).

    Raises
    ------
    AssertionError
        If the number of features in X and Y do not match.
        If `metric` is not one of the supported metrics.
        If `label_transfer` is required but not provided.
    """

    if not isinstance(X, list):
        X = [X]
    if not isinstance(Y, list):
        Y = [Y]
    if not isinstance(metric, list):
        metric = [metric]
    dist_mats = []
    for (x, y, m) in zip(X, Y, metric):
        if m == "label":
            assert label_transfer is not None, "label_transfer must be provided for metric 'label'."
            dist_mats.append(_label_distance_backend(x, y, label_transfer))
        elif m in ["euc", "euclidean"]:
            dist_mats.append(_euc_distance_backend(x, y, squared=True))
        elif m in ["square_euc", "square_euclidean"]:
            dist_mats.append(_euc_distance_backend(x, y, squared=False))
        elif m == "kl":
            dist_mats.append(
                _kl_distance_backend(
                    x,
                    y,
                )
            )
        elif m == "sym_kl":
            dist_mats.append(
                (
                    _kl_distance_backend(
                        x,
                        y,
                    )
                    + _kl_distance_backend(y, x).T
                )
                / 2
            )
        elif m in ["cos", "cosine"]:
            dist_mats.append(
                _cosine_distance_backend(
                    x,
                    y,
                )
            )

    return dist_mats


def calc_probability(
    distance_matrix: Union[np.ndarray, torch.Tensor],
    probability_type: str = "gauss",
    probability_parameter: Optional[float] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate probability based on the distance matrix and specified probability type.

    Parameters
    ----------
    distance_matrix : np.ndarray or torch.Tensor
        The distance matrix.
    probability_type : str, optional
        The type of probability to calculate. Options are 'Gauss', 'cos_prob', and 'prob'. Default is 'Gauss'.
    probability_parameter : Optional[float], optional
        The parameter for the probability calculation. Required for certain probability types. Default is None.

    Returns
    -------
    np.ndarray or torch.Tensor
        The calculated probability matrix.

    Raises
    ------
    ValueError
        If `probability_type` is not one of the supported types or if required parameters are missing.
    """

    # Get the appropriate backend (either NumPy or PyTorch)
    nx = ot.backend.get_backend(distance_matrix)

    if probability_type.lower() == "gauss":
        if probability_parameter is None:
            raise ValueError("probability_parameter must be provided for 'Gauss' probability type.")
        probability = nx.exp(-distance_matrix / (2 * probability_parameter))
    elif probability_type.lower() == "cos_prob":
        probability = distance_matrix * 0.5 + 0.5
    elif probability_type.lower() == "prob":
        probability = distance_matrix
    else:
        raise ValueError(f"Unsupported probability type: {probability_type}")

    return probability

##########################################
# Variational variables update functions #
##########################################

def update_nonrigid(
    nx,
    type_as,
    SVI_mode,
    guidance_effect,
    SigmaInv,
    step_size,
    sigma2,
    lambdaVF,
    GammaSparse,
    U,
    K_NA,
    PXB_term,
    P,
    coordsB,
    RnA,
    guidance_epsilon,
    U_I,
    R_AI,
    X_BI,
):
    if SVI_mode:
        SigmaInv = (
            step_size * (sigma2 * lambdaVF * GammaSparse + _dot(nx)(U.T, nx.einsum("ij,i->ij", U, K_NA)))
            + (1 - step_size) * SigmaInv
        )
        PXB_term = step_size * (_dot(nx)(P, coordsB) - nx.einsum("ij,i->ij", RnA, K_NA)) + (1 - step_size) * PXB_term
    else:
        SigmaInv = sigma2 * lambdaVF * GammaSparse + _dot(nx)(U.T, nx.einsum("ij,i->ij", U, K_NA))
        PXB_term = _dot(nx)(P, coordsB) - nx.einsum("ij,i->ij", RnA, K_NA)

    UPXB_term = _dot(nx)(U.T, PXB_term)

    if (guidance_effect == "nonrigid") or (guidance_effect == "both"):
        SigmaInv += (sigma2 / guidance_epsilon) * _dot(nx)(U_I.T, U_I)
        UPXB_term += (sigma2 / guidance_epsilon) * _dot(nx)(U_I.T, X_BI - R_AI)

    Sigma = _pinv(nx)(SigmaInv)
    Coff = _dot(nx)(Sigma, UPXB_term)

    VnA = _dot(nx)(U, Coff)
    V_AI = _dot(nx)(U_I, Coff)
    SigmaDiag = sigma2 * nx.einsum("ij->i", nx.einsum("ij,ji->ij", U, _dot(nx)(Sigma, U.T)))

    return VnA, V_AI, SigmaDiag, SigmaInv, PXB_term, Coff


#################################
# Kernel construction functions #
#################################


def con_K(
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    beta: Union[int, float] = 0.01,
) -> Union[np.ndarray, torch.Tensor]:
    """con_K constructs the Squared Exponential (SE) kernel, where K(i,j)=k(X_i,Y_j)=exp(-beta*||X_i-Y_j||^2).

    Args:
        X: The first vector X\in\mathbb{R}^{N\times d}
        Y: The second vector X\in\mathbb{R}^{M\times d}
        beta: The length-scale of the SE kernel.
        use_chunk (bool, optional): Whether to use chunk to reduce the GPU memory usage. Note that if set to ``True'' it will slow down the calculation. Defaults to False.

    Returns:
        K: The kernel K\in\mathbb{R}^{N\times M}
    """

    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    nx = ot.backend.get_backend(X, Y)

    [K] = calc_distance(
        X=X,
        Y=Y,
        metric="euc",
    )
    K = nx.exp(-beta * K)
    return K


def con_K_geodist():
    pass


def kl_divergence_backend(X, Y, probabilistic=True):
    """
    Returns pairwise KL divergence (over all pairs of samples) of two matrices X and Y.
    Takes advantage of POT backend to speed up computation.
    Args:
        X: np array with dim (n_samples by n_features)
        Y: np array with dim (m_samples by n_features)
    Returns:
        D: np array with dim (n_samples by m_samples). Pairwise KL divergence matrix.
    """
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    nx = ot.backend.get_backend(X, Y)
    if probabilistic:
        X = X / nx.sum(X, axis=1, keepdims=True)
        Y = Y / nx.sum(Y, axis=1, keepdims=True)
    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum("ij,ij->i", X, log_X)
    X_log_X = nx.reshape(X_log_X, (1, X_log_X.shape[0]))
    D = X_log_X.T - nx.dot(X, log_Y.T)
    return D


def kl_distance(
    X_A: Union[np.ndarray, torch.Tensor],
    X_B: Union[np.ndarray, torch.Tensor],
    use_gpu: bool = True,
    chunk_num: int = 1,
    symmetry: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """Calculate the KL distance between two vectors

    Args:
        X_A (Union[np.ndarray, torch.Tensor]): The first input vector with shape n x d
        X_B (Union[np.ndarray, torch.Tensor]): The second input vector with shape m x d
        use_gpu (bool, optional): Whether to use GPU for chunk. Defaults to True.
        chunk_num (int, optional): The number of chunks. The larger the number, the smaller the GPU memory usage, but the slower the calculation speed. Defaults to 20.
        symmetry (bool, optional): Whether to use symmetric KL divergence. Defaults to True.

    Returns:
        Union[np.ndarray, torch.Tensor]: KL distance matrix of two vectors with shape n x m.
    """
    nx = ot.backend.get_backend(X_A, X_B)
    data_on_gpu = False
    if nx_torch(nx):
        if X_A.is_cuda:
            data_on_gpu = True
    type_as = X_A[0, 0].cpu() if nx_torch(nx) else X_A[0, 0]
    use_gpu = True if use_gpu and nx_torch(nx) and torch.cuda.is_available() else False
    chunk_flag = False
    # Probabilistic normalization
    X_A = X_A / nx.sum(X_A, axis=1, keepdims=True)
    X_B = X_B / nx.sum(X_B, axis=1, keepdims=True)
    while True:
        try:
            if chunk_num == 1:
                if symmetry:
                    DistMat = (kl_divergence_backend(X_A, X_B, False) + kl_divergence_backend(X_B, X_A, False).T) / 2
                else:
                    DistMat = kl_divergence_backend(X_A, X_B, False)
                break
            else:
                # convert to numpy to save the GPU memory
                if chunk_flag == False:
                    X_A, X_B = nx.to_numpy(X_A), nx.to_numpy(X_B)
                chunk_flag = True
                # chunk
                X_As = np.array_split(X_A, chunk_num, axis=0)
                X_Bs = np.array_split(X_B, chunk_num, axis=0)
                arr = []  # array for temporary storage of results
                for x_As in X_As:
                    arr2 = []  # array for temporary storage of results
                    for x_Bs in X_Bs:
                        if use_gpu:
                            if symmetry:
                                arr2.append(
                                    (
                                        kl_divergence_backend(
                                            nx.from_numpy(x_As, type_as=type_as).cuda(),
                                            nx.from_numpy(x_Bs, type_as=type_as).cuda(),
                                            False,
                                        ).cpu()
                                        + kl_divergence_backend(
                                            nx.from_numpy(x_Bs, type_as=type_as).cuda(),
                                            nx.from_numpy(x_As, type_as=type_as).cuda(),
                                            False,
                                        )
                                        .cpu()
                                        .T
                                    )
                                    / 2
                                )
                            else:
                                arr2.append(
                                    kl_divergence_backend(
                                        nx.from_numpy(x_As, type_as=type_as).cuda(),
                                        nx.from_numpy(x_Bs, type_as=type_as).cuda(),
                                        False,
                                    ).cpu()
                                )
                        else:
                            if symmetry:
                                arr2.append(
                                    nx.to_numpy(
                                        kl_divergence_backend(
                                            nx.from_numpy(x_As, type_as=type_as),
                                            nx.from_numpy(x_Bs, type_as=type_as),
                                            False,
                                        )
                                        + kl_divergence_backend(
                                            nx.from_numpy(x_Bs, type_as=type_as),
                                            nx.from_numpy(x_As, type_as=type_as),
                                            False,
                                        ).T
                                    )
                                    / 2
                                )
                            else:
                                arr2.append(
                                    kl_divergence_backend(
                                        nx.from_numpy(x_As, type_as=type_as),
                                        nx.from_numpy(x_Bs, type_as=type_as),
                                        False,
                                    )
                                )
                    arr.append(nx.concatenate(arr2, axis=1))
                DistMat = nx.concatenate(arr, axis=0)
                break
        except:
            chunk_num = chunk_num * 2
            print("kl chunk more")
    if data_on_gpu and chunk_num != 1:
        DistMat = DistMat.cuda()
    return DistMat


def calc_exp_dissimilarity(
    X_A: Union[np.ndarray, torch.Tensor],
    X_B: Union[np.ndarray, torch.Tensor],
    dissimilarity: str = "kl",
    chunk_num: int = 1,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate expression dissimilarity.
    Args:
        X_A: Gene expression matrix of sample A.
        X_B: Gene expression matrix of sample B.
        dissimilarity: Expression dissimilarity measure: ``'kl'``, ``'euclidean'``, ``'euc'``, ``'cos'``, or ``'cosine'``.

    Returns:
        Union[np.ndarray, torch.Tensor]: The dissimilarity matrix of two feature samples.
    """
    nx = ot.backend.get_backend(X_A, X_B)

    assert dissimilarity in [
        "kl",
        "euclidean",
        "euc",
        "cos",
        "cosine",
    ], "``dissimilarity`` value is wrong. Available ``dissimilarity`` are: ``'kl'``, ``'euclidean'``, ``'euc'``, ``'cos'``, and ``'cosine'``."
    if dissimilarity.lower() == "kl":
        X_A = X_A + 0.01
        X_B = X_B + 0.01
        X_A = X_A / nx.sum(X_A, axis=1, keepdims=True)
        X_B = X_B / nx.sum(X_B, axis=1, keepdims=True)
    while True:
        try:
            if chunk_num == 1:
                DistMat = _dist(X_A, X_B, dissimilarity)
                break
            else:
                X_As = _chunk(nx, X_A, chunk_num, 0)
                X_Bs = _chunk(nx, X_B, chunk_num, 0)
                arr = []  # array for temporary storage of results
                for x_As in X_As:
                    arr2 = []
                    for x_Bs in X_Bs:
                        arr2.append(_dist(x_As, x_Bs, dissimilarity))
                    arr.append(nx.concatenate(arr2, axis=1))
                DistMat = nx.concatenate(arr, axis=0)
                break
        except:
            chunk_num = chunk_num * 2
            print("chunk more")
    return DistMat


def cal_dist(
    X_A: Union[np.ndarray, torch.Tensor],
    X_B: Union[np.ndarray, torch.Tensor],
    use_gpu: bool = True,
    chunk_num: int = 1,
    return_gpu: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """Calculate the distance between two vectors

    Args:
        X_A (Union[np.ndarray, torch.Tensor]): The first input vector with shape n x d
        X_B (Union[np.ndarray, torch.Tensor]): The second input vector with shape m x d
        use_gpu (bool, optional): Whether to use GPU for chunk. Defaults to True.
        chunk_num (int, optional): The number of chunks. The larger the number, the smaller the GPU memory usage, but the slower the calculation speed. Defaults to 1.

    Returns:
        Union[np.ndarray, torch.Tensor]: Distance matrix of two vectors with shape n x m.
    """
    nx = ot.backend.get_backend(X_A, X_B)
    data_on_gpu = False
    if nx_torch(nx):
        if X_A.is_cuda:
            data_on_gpu = True
    type_as = X_A[0, 0].cpu() if nx_torch(nx) else X_A[0, 0]
    use_gpu = True if use_gpu and nx_torch(nx) and torch.cuda.is_available() else False
    chunk_flag = False
    while True:
        try:
            if chunk_num == 1:
                DistMat = _dist(X_A, X_B, "euc")
                break
            else:
                # convert to numpy to save the GPU memory
                if chunk_flag == False:
                    X_A, X_B = nx.to_numpy(X_A), nx.to_numpy(X_B)
                chunk_flag = True
                # chunk
                X_As = np.array_split(X_A, chunk_num, axis=0)
                X_Bs = np.array_split(X_B, chunk_num, axis=0)
                arr = []  # array for temporary storage of results
                for x_As in X_As:
                    arr2 = []  # array for temporary storage of results
                    for x_Bs in X_Bs:
                        if use_gpu:
                            arr2.append(
                                ot.dist(
                                    nx.from_numpy(x_As, type_as=type_as).cuda(),
                                    nx.from_numpy(x_Bs, type_as=type_as).cuda(),
                                ).cpu()
                            )
                        else:
                            arr2.append(
                                ot.dist(
                                    nx.from_numpy(x_As, type_as=type_as),
                                    nx.from_numpy(x_Bs, type_as=type_as),
                                )
                            )
                    arr.append(nx.concatenate(arr2, axis=1))
                DistMat = nx.concatenate(arr, axis=0)  # not convert to GPU
                break
        except:
            chunk_num = chunk_num * 2
            print("dist chunk more")
    if data_on_gpu and chunk_num != 1 and return_gpu:
        DistMat = DistMat.cuda()
    return DistMat


def cal_dot(
    mat1: Union[np.ndarray, torch.Tensor],
    mat2: Union[np.ndarray, torch.Tensor],
    use_chunk: bool = False,
    use_gpu: bool = True,
    chunk_num: int = 20,
) -> Union[np.ndarray, torch.Tensor]:
    """Calculate the matrix multiplication of two matrices

    Args:
        mat1 (Union[np.ndarray, torch.Tensor]): The first input matrix with shape n x d
        mat2 (Union[np.ndarray, torch.Tensor]): The second input matrix with shape d x m. We suppose m << n and does not require chunk.
        use_chunk (bool, optional): Whether to use chunk to reduce the GPU memory usage. Note that if set to ``True'' it will slow down the calculation. Defaults to False.
        use_gpu (bool, optional): Whether to use GPU for chunk. Defaults to True.
        chunk_num (int, optional): The number of chunks. The larger the number, the smaller the GPU memory usage, but the slower the calculation speed. Defaults to 20.

    Returns:
        Union[np.ndarray, torch.Tensor]: Matrix multiplication result with shape n x m
    """
    nx = ot.backend.get_backend(mat1, mat2)
    type_as = mat1[0, 0]
    use_gpu = True if use_gpu and nx_torch(nx) and torch.cuda.is_available() else False
    if not use_chunk:
        Mat = _dot(nx)(mat1, mat2)
        return Mat
    else:
        # convert to numpy to save the GPU memory
        mat1 = nx.to_numpy(mat1)
        if use_gpu:
            mat2 = mat2.cuda()
        # chunk
        mat1s = np.array_split(mat1, chunk_num, axis=0)
        arr = []  # array for temporary storage of results
        for mat1ss in mat1s:
            if use_gpu:
                arr.append(_dot(nx)(nx.from_numpy(mat1ss, type_as=type_as).cuda(), mat2).cpu())
            else:
                arr.append(_dot(nx)(nx.from_numpy(mat1ss, type_as=type_as), mat2))
        Mat = nx.concatenate(arr, axis=0)
        return Mat


def get_optimal_R(
    coordsA: Union[np.ndarray, torch.Tensor],
    coordsB: Union[np.ndarray, torch.Tensor],
    P: Union[np.ndarray, torch.Tensor],
    R_init: Union[np.ndarray, torch.Tensor],
):
    """Get the optimal rotation matrix R

    Args:
        coordsA (Union[np.ndarray, torch.Tensor]): The first input matrix with shape n x d
        coordsB (Union[np.ndarray, torch.Tensor]): The second input matrix with shape n x d
        P (Union[np.ndarray, torch.Tensor]): The optimal transport matrix with shape n x n

    Returns:
        Union[np.ndarray, torch.Tensor]: The optimal rotation matrix R with shape d x d
    """
    nx = ot.backend.get_backend(coordsA, coordsB, P, R_init)
    NA, NB, D = coordsA.shape[0], coordsB.shape[0], coordsA.shape[1]
    Sp = nx.einsum("ij->", P)
    K_NA = nx.einsum("ij->i", P)
    K_NB = nx.einsum("ij->j", P)
    VnA = nx.zeros(coordsA.shape, type_as=coordsA[0, 0])
    mu_XnA, mu_VnA, mu_XnB = (
        _dot(nx)(K_NA, coordsA) / Sp,
        _dot(nx)(K_NA, VnA) / Sp,
        _dot(nx)(K_NB, coordsB) / Sp,
    )
    XnABar, VnABar, XnBBar = coordsA - mu_XnA, VnA - mu_VnA, coordsB - mu_XnB
    A = -_dot(nx)(nx.einsum("ij,i->ij", VnABar, K_NA).T - _dot(nx)(P, XnBBar).T, XnABar)

    # get the optimal rotation matrix R
    svdU, svdS, svdV = _linalg(nx).svd(A)
    C = _identity(nx, D, type_as=coordsA[0, 0])
    C[-1, -1] = _linalg(nx).det(_dot(nx)(svdU, svdV))
    R = _dot(nx)(_dot(nx)(svdU, C), svdV)
    t = mu_XnB - mu_VnA - _dot(nx)(mu_XnA, R.T)
    optimal_RnA = _dot(nx)(coordsA, R.T) + t
    return optimal_RnA, R, t


###############################
# Distance Matrix Calculation #
###############################
def _cal_cosine_similarity(tensor1, tensor2, dim=1, eps=1e-8):
    tensor1_norm = torch.sqrt(torch.sum(tensor1**2, dim=dim, keepdim=True))
    tensor2_norm = torch.sqrt(torch.sum(tensor2**2, dim=dim, keepdim=True))
    tensor1_norm = torch.clamp(tensor1_norm, min=eps)
    tensor2_norm = torch.clamp(tensor2_norm, min=eps)
    dot_product = torch.sum(tensor1 * tensor2, dim=dim, keepdim=True)
    cosine_similarity = dot_product / (tensor1_norm * tensor2_norm)
    cosine_similarity = cosine_similarity.squeeze(dim)
    return cosine_similarity


def _cos_similarity(
    mat1: Union[np.ndarray, torch.Tensor],
    mat2: Union[np.ndarray, torch.Tensor],
):
    nx = ot.backend.get_backend(mat1, mat2)
    if nx_torch(nx):
        mat1_unsqueeze = mat1.unsqueeze(-1)
        mat2_unsqueeze = mat2.unsqueeze(-1).transpose(0, 2)
        distMat = _cal_cosine_similarity(mat1_unsqueeze, mat2_unsqueeze) * 0.5 + 0.5
    else:
        distMat = (-ot.dist(mat1, mat2, metric="cosine") + 1) * 0.5 + 0.5
    return distMat


def _dist(
    mat1: Union[np.ndarray, torch.Tensor],
    mat2: Union[np.ndarray, torch.Tensor],
    metric: str = "euc",
) -> Union[np.ndarray, torch.Tensor]:
    assert metric in [
        "euc",
        "euclidean",
        "kl",
        "cos",
        "cosine",
    ], "``metric`` value is wrong. Available ``metric`` are: ``'euc'``, ``'euclidean'`` and ``'kl'``."
    nx = ot.backend.get_backend(mat1, mat2)
    if metric.lower() == "euc" or metric.lower() == "euclidean":
        distMat = nx.sum(mat1**2, 1)[:, None] + nx.sum(mat2**2, 1)[None, :] - 2 * _dot(nx)(mat1, mat2.T)
    elif metric.lower() == "kl":
        distMat = (
            nx.sum(mat1 * nx.log(mat1), 1)[:, None]
            + nx.sum(mat2 * nx.log(mat2), 1)[None, :]
            - _dot(nx)(mat1, nx.log(mat2).T)
            - _dot(nx)(mat2, nx.log(mat1).T).T
        ) / 2
    elif (metric.lower() == "cosine") or (metric.lower() == "cos"):
        distMat = _cos_similarity(mat1, mat2)
    return distMat


def voxel_data(
    nx: Union[ot.backend.TorchBackend, ot.backend.NumpyBackend],
    coords: Union[np.ndarray, torch.Tensor],
    gene_exp: Union[np.ndarray, torch.Tensor],
    voxel_size: Optional[float] = None,
    voxel_num: Optional[int] = 10000,
):
    """
    Voxelization of the data.
    Parameters
    ----------
    coords: np.ndarray or torch.Tensor
        The coordinates of the data points.
    gene_exp: np.ndarray or torch.Tensor
        The gene expression of the data points.
    voxel_size: float
        The size of the voxel.
    voxel_num: int
        The number of voxels.
    Returns
    -------
    voxel_coords: np.ndarray or torch.Tensor
        The coordinates of the voxels.
    voxel_gene_exp: np.ndarray or torch.Tensor
        The gene expression of the voxels.
    """
    # nx = ot.backend.get_backend(coords, gene_exp)
    N, D = coords.shape[0], coords.shape[1]
    coords = nx.to_numpy(coords)
    gene_exp = nx.to_numpy(gene_exp)

    # create the voxel grid
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    if voxel_size is None:
        voxel_size = np.sqrt(np.prod(max_coords - min_coords)) / (np.sqrt(N) / 5)
        # print(voxel_size)
    voxel_steps = (max_coords - min_coords) / int(np.sqrt(voxel_num))
    voxel_coords = [
        np.arange(min_coord, max_coord, voxel_step)
        for min_coord, max_coord, voxel_step in zip(min_coords, max_coords, voxel_steps)
    ]
    voxel_coords = np.stack(np.meshgrid(*voxel_coords), axis=-1).reshape(-1, D)
    voxel_gene_exps = np.zeros((voxel_coords.shape[0], gene_exp.shape[1]))
    is_voxels = np.zeros((voxel_coords.shape[0],))
    # assign the data points to the voxels
    for i, voxel_coord in enumerate(voxel_coords):
        dists = np.sqrt(np.sum((coords - voxel_coord) ** 2, axis=1))
        mask = dists < voxel_size / 2
        if np.any(mask):
            voxel_gene_exps[i] = np.mean(gene_exp[mask], axis=0)
            is_voxels[i] = 1
    voxel_coords = voxel_coords[is_voxels == 1, :]
    voxel_gene_exps = voxel_gene_exps[is_voxels == 1, :]
    return voxel_coords, voxel_gene_exps

#################################
# Funcs between Numpy and Torch #
#################################


# Empty cache
def empty_cache(device: str = "cpu"):
    if device != "cpu":
        torch.cuda.empty_cache()


# Check if nx is a torch backend
nx_torch = lambda nx: True if isinstance(nx, ot.backend.TorchBackend) else False

# Concatenate expression matrices
_cat = lambda nx, x, dim: torch.cat(x, dim=dim) if nx_torch(nx) else np.concatenate(x, axis=dim)
_unique = lambda nx, x, dim: torch.unique(x, dim=dim) if nx_torch(nx) else np.unique(x, axis=dim)
_var = lambda nx, x, dim: torch.var(x, dim=dim) if nx_torch(nx) else np.var(x, axis=dim)

_data = (
    lambda nx, data, type_as: torch.tensor(data, device=type_as.device, dtype=type_as.dtype)
    if nx_torch(nx)
    else np.asarray(data, dtype=type_as.dtype)
)
_unsqueeze = lambda nx: torch.unsqueeze if nx_torch(nx) else np.expand_dims
_mul = lambda nx: torch.multiply if nx_torch(nx) else np.multiply
_power = lambda nx: torch.pow if nx_torch(nx) else np.power
_psi = lambda nx: torch.special.psi if nx_torch(nx) else psi
_pinv = lambda nx: torch.linalg.pinv if nx_torch(nx) else pinv
_dot = lambda nx: torch.matmul if nx_torch(nx) else np.dot
_identity = (
    lambda nx, N, type_as: torch.eye(N, dtype=type_as.dtype, device=type_as.device)
    if nx_torch(nx)
    else np.identity(N, dtype=type_as.dtype)
)
_linalg = lambda nx: torch.linalg if nx_torch(nx) else np.linalg
_prod = lambda nx: torch.prod if nx_torch(nx) else np.prod
_pi = lambda nx: torch.pi if nx_torch(nx) else np.pi
_chunk = (
    lambda nx, x, chunk_num, dim: torch.chunk(x, chunk_num, dim=dim)
    if nx_torch(nx)
    else np.array_split(x, chunk_num, axis=dim)
)
_randperm = lambda nx: torch.randperm if nx_torch(nx) else np.random.permutation
_roll = lambda nx: torch.roll if nx_torch(nx) else np.roll
_choice = (
    lambda nx, length, size: torch.randperm(length)[:size]
    if nx_torch(nx)
    else np.random.choice(length, size, replace=False)
)
_topk = (
    lambda nx, x, topk, axis: torch.topk(x, topk, dim=axis)[1] if nx_torch(nx) else np.argpartition(x, topk, axis=axis)
)
_dstack = lambda nx: torch.dstack if nx_torch(nx) else np.dstack
_vstack = lambda nx: torch.vstack if nx_torch(nx) else np.vstack
_hstack = lambda nx: torch.hstack if nx_torch(nx) else np.hstack

_split = (
    lambda nx, x, chunk_size, dim: torch.split(x, chunk_size, dim)
    if nx_torch(nx)
    else np.array_split(x, chunk_size, axis=dim)
)


def torch_like_split(arr, size, dim=0):
    if dim < 0:
        dim += arr.ndim
    shape = arr.shape
    arr = np.swapaxes(arr, dim, -1)
    flat_arr = arr.reshape(-1, shape[dim])
    num_splits = flat_arr.shape[-1] // size
    remainder = flat_arr.shape[-1] % size
    splits = np.array_split(flat_arr[:, : num_splits * size], num_splits, axis=-1)
    if remainder:
        splits.append(flat_arr[:, num_splits * size :])
    splits = [np.swapaxes(split.reshape(*shape[:dim], -1, *shape[dim + 1 :]), dim, -1) for split in splits]

    return splits
