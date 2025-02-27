import os
from functools import partial
from multiprocessing import Pool
import numpy as np
from scipy import stats
from scipy.spatial.distance import squareform
from tqdm import tqdm


def compute_per_subject_spearman(permutation_idx, layer_rdm, roi_rdm):
    square_layer_rdm = squareform(layer_rdm)
    permutation = squareform(square_layer_rdm[permutation_idx, :], checks=False)
    null_subject_values = []
    for i, subject_roi_rdm in enumerate(roi_rdm):
        vector_subject_roi_rdm = squareform(subject_roi_rdm, checks=False)
        null_subject_values.append(stats.spearmanr(permutation, vector_subject_roi_rdm).statistic)
    return null_subject_values


def compute_null_distribution(model_rdms_path, brain_rdms_path, null_distrib_path, skips, permutations_idx, n_jobs=8):
    if os.path.exists(null_distrib_path):
        old_null_distribs = np.load(null_distrib_path)
    else:
        old_null_distribs = {}
    brain_rdms_files = [f for f in os.listdir(brain_rdms_path) if f.endswith(".npz")]
    model_rdms_files = [f for f in os.listdir(model_rdms_path) if f.endswith(".npz")]
    null_distribs = {}
    for roi_rdm_file in tqdm(brain_rdms_files):
        for layer_rdm_file in tqdm(model_rdms_files):
            layer_name = layer_rdm_file.split("RDM_")[1].split(".npz")[0]
            if layer_name in skips:
                continue
            key = f"{layer_name}_{roi_rdm_file.split('.npz')[0]}"
            if key in old_null_distribs:
                null_distribs[key] = old_null_distribs[key]
                continue
            roi_rdm = np.load(os.path.join(brain_rdms_path, roi_rdm_file))["arr_0"]
            layer_rdm = np.load(os.path.join(model_rdms_path, layer_rdm_file))["rdm"]
            with Pool(n_jobs) as p:
                null_distrib = p.map(
                    partial(compute_per_subject_spearman, layer_rdm=layer_rdm, roi_rdm=roi_rdm), permutations_idx
                )
            null_distrib = np.array(null_distrib)
            null_distribs[key] = null_distrib.mean(axis=1)
    np.savez(null_distrib_path, **null_distribs)
    return null_distribs


def compute_significance_against_zero(null_distrib, observed_statistic):
    pvalue = np.sum(np.abs(null_distrib) >= np.abs(observed_statistic)) / len(null_distrib)
    return pvalue


def compute_pairwise_significance(null_distrib_1, null_distrib_2, observed_statistic_1, observed_statistic_2):
    true_mean_diff = observed_statistic_1 - observed_statistic_2
    null_distrib_mean_diff = null_distrib_1 - null_distrib_2
    pvalue = compute_significance_against_zero(null_distrib_mean_diff, true_mean_diff)
    return pvalue
