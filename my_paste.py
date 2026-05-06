from typing import List, Optional, Tuple, Union

import numpy as np
import ot
import torch
from anndata import AnnData
from sklearn.decomposition import NMF
from typing import List, Optional, Tuple, Union
from scipy.spatial import cKDTree

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from spateo_utils import align_preprocess, calc_exp_dissimilarity


def get_optimal_mapping_relationship(
    X: np.ndarray,
    Y: np.ndarray,
    pi: np.ndarray,
    keep_all: bool = False,
):
    X_max_index = np.argwhere((pi.T == pi.T.max(axis=0)).T)
    Y_max_index = np.argwhere(pi == pi.max(axis=0))
    if not keep_all:
        values, counts = np.unique(X_max_index[:, 0], return_counts=True)
        x_index_unique, x_index_repeat = values[counts == 1], values[counts != 1]
        X_max_index_unique = X_max_index[np.isin(X_max_index[:, 0], x_index_unique)]

        for i in x_index_repeat:
            i_max_index = X_max_index[X_max_index[:, 0] == i]
            i_kdtree = cKDTree(Y[i_max_index[:, 1]])
            _, ii = i_kdtree.query(X[i], k=1)
            X_max_index_unique = np.concatenate([X_max_index_unique, i_max_index[ii].reshape(1, 2)], axis=0)

        values, counts = np.unique(Y_max_index[:, 1], return_counts=True)
        y_index_unique, y_index_repeat = values[counts == 1], values[counts != 1]
        Y_max_index_unique = Y_max_index[np.isin(Y_max_index[:, 1], y_index_unique)]

        for i in y_index_repeat:
            i_max_index = Y_max_index[Y_max_index[:, 1] == i]
            i_kdtree = cKDTree(X[i_max_index[:, 0]])
            _, ii = i_kdtree.query(Y[i], k=1)
            Y_max_index_unique = np.concatenate([Y_max_index_unique, i_max_index[ii].reshape(1, 2)], axis=0)

        X_max_index = X_max_index_unique.copy()
        Y_max_index = Y_max_index_unique.copy()

    X_pi_value = pi[X_max_index[:, 0], X_max_index[:, 1]].reshape(-1, 1)
    Y_pi_value = pi[Y_max_index[:, 0], Y_max_index[:, 1]].reshape(-1, 1)
    return X_max_index, X_pi_value, Y_max_index, Y_pi_value


def cell_directions(
    adataA: AnnData,
    adataB: AnnData,
    layer: str = "X",
    genes: Optional[Union[list, np.ndarray]] = None,
    spatial_key: str = "align_spatial",
    key_added: str = "mapping",
    alpha: float = 0.001,
    numItermax: int = 200,
    numItermaxEmd: int = 100000,
    dtype: str = "float32",
    device: str = "cpu",
    keep_all: bool = False,
    inplace: bool = True,
    **kwargs,
) -> Tuple[Optional[AnnData], np.ndarray]:
    """
    Obtain the optimal mapping relationship and developmental direction between cells for samples between continuous developmental stages.

    Args:
        adataA: AnnData object of sample A from continuous developmental stages.
        adataB: AnnData object of sample B from continuous developmental stages.
        layer: If ``'X'``, uses ``.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``.layers[layer]``.
        genes: Genes used for calculation. If None, use all common genes for calculation.
        spatial_key: The key in ``.obsm`` that corresponds to the spatial coordinate of each cell.
        The key that will be used for the vector field key in ``.uns``.
        key_added: The key that will be used in ``.obsm``.

                  * ``X_{key_added}``-The ``X_{key_added}`` that will be used for the coordinates of the cell that maps optimally in the next stage.
                  * ``V_{key_added}``-The ``V_{key_added}`` that will be used for the cell developmental directions.
        alpha: Alignment tuning parameter. Note: 0 <= alpha <= 1.

               When ``alpha = 0`` only the gene expression data is taken into account,
               while when ``alpha =1`` only the spatial coordinates are taken into account.
        numItermax: Max number of iterations for cg during FGW-OT.
        numItermaxEmd: Max number of iterations for emd during FGW-OT.
        dtype: The floating-point number type. Only ``float32`` and ``float64``.
        device: Equipment used to run the program. You can also set the specified GPU for running. ``E.g.: '0'``
        keep_all: Whether to retain all the optimal relationships obtained only based on the pi matrix, If ``keep_all``
                  is False, the optimal relationships obtained based on the pi matrix and the nearest coordinates.
        inplace: Whether to copy adata or modify it inplace.
        **kwargs: Additional parameters that will be passed to ``pairwise_align`` function.

    Returns:
        An ``AnnData`` object of sample A is updated/copied with the ``X_{key_added}`` and ``V_{key_added}`` in the ``.obsm`` attribute.
        A pi metrix.
    """
    # Calculate and returns optimal alignment of two models.
    pi, _ = paste_pairwise_align(
        sampleA=adataA.copy(),
        sampleB=adataB.copy(),
        spatial_key=spatial_key,
        layer=layer,
        genes=genes,
        alpha=alpha,
        numItermax=numItermax,
        numItermaxEmd=numItermaxEmd,
        dtype=dtype,
        device=device,
        dissimilarity="euclidean",
        **kwargs,
    )

    max_index, pi_value, _, _ = get_optimal_mapping_relationship(
        X=adataA.obsm[spatial_key].copy(), Y=adataB.obsm[spatial_key].copy(), pi=pi, keep_all=keep_all
    )

    mapping_data = pd.DataFrame(
        np.concatenate([max_index, pi_value], axis=1),
        columns=["index_x", "index_y", "pi_value"],
    ).astype(
        dtype={
            "index_x": np.int32,
            "index_y": np.int32,
            "pi_value": np.float64,
        }
    )
    mapping_data.sort_values(by=["index_x", "pi_value"], ascending=[True, False], inplace=True)
    mapping_data.drop_duplicates(subset=["index_x"], keep="first", inplace=True)

    adataA.obsm[f"X_{key_added}"] = adataB.obsm[spatial_key][mapping_data["index_y"].values]
    adataA.obsm[f"V_{key_added}"] = adataA.obsm[f"X_{key_added}"] - adataA.obsm[spatial_key]

    return mapping_data if inplace else adataA, pi, mapping_data

def paste_pairwise_align(
    sampleA: AnnData,
    sampleB: AnnData,
    layer: str = "X",
    genes: Optional[Union[list, np.ndarray]] = None,
    spatial_key: str = "spatial",
    alpha: float = 0.1,
    dissimilarity: str = "kl",
    G_init=None,
    a_distribution=None,
    b_distribution=None,
    norm: bool = False,
    numItermax: int = 200,
    numItermaxEmd: int = 100000,
    dtype: str = "float32",
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[np.ndarray, Optional[int]]:
    """
    Calculates and returns optimal alignment of two slices.

    Args:
        sampleA: Sample A to align.
        sampleB: Sample B to align.
        layer: If `'X'`, uses ``sample.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``sample.layers[layer]``.
        genes: Genes used for calculation. If None, use all common genes for calculation.
        spatial_key: The key in `.obsm` that corresponds to the raw spatial coordinates.
        alpha:  Alignment tuning parameter. Note: 0 <= alpha <= 1.
                When α = 0 only the gene expression data is taken into account,
                while when α =1 only the spatial coordinates are taken into account.
        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
        G_init: Initial mapping to be used in FGW-OT, otherwise default is uniform mapping.
        a_distribution: Distribution of sampleA spots, otherwise default is uniform.
        b_distribution: Distribution of sampleB spots, otherwise default is uniform.
        norm: If ``True``, scales spatial distances such that neighboring spots are at distance 1. Otherwise, spatial distances remain unchanged.
        numItermax: Max number of iterations for cg during FGW-OT.
        numItermaxEmd: Max number of iterations for emd during FGW-OT.
        dtype: The floating-point number type. Only float32 and float64.
        device: Equipment used to run the program. You can also set the specified GPU for running. E.g.: '0'.
        verbose: If ``True``, print progress updates.

    Returns:
        pi: Alignment of spots.
        obj: Objective function output of FGW-OT.
    """

    # Preprocessing
    (nx, type_as, exp_matrices, spatial_coords, label_transfer,normalize_scale, normalize_mean_list, common_genes,) = align_preprocess(
        samples=[sampleA, sampleB],
        genes=genes,
        spatial_key=spatial_key,
        rep_layer=layer,
        rep_field='obsm',
        normalize_c=True,
        normalize_g=False,
        dtype=dtype,
        device=device,
        verbose=verbose,
    )

    # Calculate spatial distances
    coordsA, coordsB = spatial_coords[0], spatial_coords[1]
    D_A = ot.dist(coordsA, coordsA, metric="euclidean")
    D_B = ot.dist(coordsB, coordsB, metric="euclidean")

    # Calculate expression dissimilarity
    X_A, X_B = exp_matrices[0], exp_matrices[1]
    M = calc_exp_dissimilarity(X_A=X_A[0], X_B=X_B[0], dissimilarity=dissimilarity)

    # init distributions
    a = np.ones((sampleA.shape[0],)) / sampleA.shape[0] if a_distribution is None else np.asarray(a_distribution)
    b = np.ones((sampleB.shape[0],)) / sampleB.shape[0] if b_distribution is None else np.asarray(b_distribution)
    a = nx.from_numpy(a, type_as=type_as)
    b = nx.from_numpy(b, type_as=type_as)

    if norm:
        D_A /= nx.min(D_A[D_A > 0])
        D_B /= nx.min(D_B[D_B > 0])

    # Run OT
    constC, hC1, hC2 = ot.gromov.init_matrix(D_A, D_B, a, b, "square_loss")

    if G_init is None:
        G0 = a[:, None] * b[None, :]
    else:
        G_init = nx.from_numpy(G_init, type_as=type_as)
        G0 = (1 / nx.sum(G_init)) * G_init

    try:
        from ot.optim import cg
    except ImportError:
        from ot.gromov import cg

    pi, log = ot.optim.cg(
        a,
        b,
        (1 - alpha) * M,
        alpha,
        lambda G: ot.gromov.gwloss(constC, hC1, hC2, G),
        lambda G: ot.gromov.gwggrad(constC, hC1, hC2, G),
        G0,
        armijo=False,
        C1=D_A,
        C2=D_B,
        constC=constC,
        numItermax=numItermax,
        numItermaxEmd=numItermaxEmd,
        log=True,
        verbose=True,
    )

    pi = nx.to_numpy(pi)
    obj = nx.to_numpy(log["loss"][-1])
    if device != "cpu":
        torch.cuda.empty_cache()

    return pi, obj