import os
import re

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import numpy as np
import pandas as pd

from net2brain.feature_extraction import FeatureExtractor
from net2brain.rdm_creation import RDMCreator
from net2brain.evaluations.rsa import RSA
from net2brain.evaluations.plotting import Plotting

from repralign.models.custom_extraction_functions import fixed_cleaner, generic_cleaner_tuples
from repralign.utils.statistical_significance import compute_null_distribution, compute_significance_against_zero
from repralign.utils.custom_plotting import (
    bmd_plot_per_roi_group,
    bmd_plot_all_layers_line_roi_groups,
)


@hydra.main(version_base=None, config_path="../configs", config_name="calc_alignment")
def main(cfg: DictConfig) -> None:

    reader = instantiate(cfg.dataset)
    dataset_path = reader.dataset_path
    stimuli_path = reader.stimuli_path
    roi_path = reader.roi_path
    selection = reader.selection

    num_rows = len(os.listdir(stimuli_path))
    permutations_idx = [np.random.permutation(num_rows) for _ in range(cfg.n_permutations)]
    all_models_null_distribs = {}
    dfs = []
    for model in cfg.models:
        model_path = os.path.join(dataset_path, cfg.models_folder, model.name, selection)
        results_csv_path = os.path.join(
            dataset_path,
            "RSA_results",
            selection,
            cfg.models_folder,
            model.name,
            "results.csv",
        )
        if not os.path.exists(os.path.dirname(results_csv_path)):
            os.makedirs(os.path.dirname(results_csv_path))
        if os.path.exists(results_csv_path):
            print(f"Results for {model.name} already exist, skipping ...")
            df = pd.read_csv(results_csv_path)
            df["ModelType"] = model.type
            if cfg.other_folders_compare:
                df["ModelFolder"] = cfg.models_folder
                df["ModelTypeFolder"] = model.type + "-" + cfg.models_folder
                for comp_folder in cfg.other_folders_compare:
                    other_results_path = os.path.join(
                        dataset_path,
                        "RSA_results",
                        selection,
                        comp_folder,
                        model.name,
                        "results.csv",
                    )
                    if not os.path.exists(other_results_path):
                        raise ValueError(
                            f"Cannot compare results with folder {comp_folder} as its results do not exist."
                        )
                    other_df = pd.read_csv(other_results_path)
                    other_df["ModelType"] = model.type
                    other_df["ModelFolder"] = comp_folder
                    other_df["ModelTypeFolder"] = model.type + "-" + comp_folder
                    dfs.append(other_df)
            dfs.append(df)
            null_distrib_path = os.path.join(model_path, "null_distrib.npz")
            null_distribs = np.load(null_distrib_path)
            all_models_null_distribs[model.name] = null_distribs
            continue
        if model.set is not None:
            fx = FeatureExtractor(
                model=model.name,
                netset=model.set,
                device=cfg.device,
                feature_cleaner=fixed_cleaner if model.set == "Pyvideo" else generic_cleaner_tuples,
            )
        else:
            model_obj = instantiate(model.cfg).to(cfg.device)
            preprocessor = (
                instantiate(model.preprocessor)
                if ("preprocessor" in model and model.preprocessor is not None)
                else None
            )
            extractor = instantiate(model.extractor) if ("extractor" in model and model.extractor is not None) else None
            fx = FeatureExtractor(
                model=model_obj,
                netset_fallback=model.netset_fallback,
                device=cfg.device,
                preprocessor=preprocessor,
                extraction_function=extractor,
                feature_cleaner=(fixed_cleaner if model.netset_fallback == "Pyvideo" else generic_cleaner_tuples),
            )
        skips = model.rsa_skips if "rsa_skips" in model else []
        feat_path = os.path.join(model_path, "feat")
        if not os.path.exists(feat_path):
            print(f"Extracting features from {model.name} ...")
            fx.extract(
                data_path=stimuli_path,
                save_path=feat_path,
                consolidate_per_layer=False,
                layers_to_extract=model.layers if "layers" in model else None,
            )
        rdm_path = os.path.join(model_path, "rdm")
        if not os.path.exists(rdm_path):
            print(f"Creating RDMs for {model.name} ...")
            creator = RDMCreator(device=cfg.device, verbose=True)
            creator.create_rdms(
                feature_path=feat_path,
                save_path=rdm_path,
                save_format="npz",
                standardize_on_dim=None,
                dim_reduction="pca",
                n_samples_estim=102,
                n_components=100,
            )
        print(f"Computing RSA for {model.name} ...")
        evaluation = RSA(
            model_rdms_path=rdm_path,
            brain_rdms_path=roi_path,
            model_name=model.name,
            skips=skips,
        )
        df = evaluation.evaluate()
        df["ROI"] = df["ROI"].apply(lambda roi: roi.split("fmri_")[1] if "fmri_" in roi else roi)
        df["Layer"] = (
            df["Layer"]
            .apply(lambda lay: lay.split("RDM_")[1] if "RDM_" in lay else lay)
            .apply(lambda lay: lay.split(".npz")[0] if ".npz" in lay else lay)
        )
        df.drop(columns=["Significance"], inplace=True)
        null_distrib_path = os.path.join(model_path, "null_distrib.npz")
        if not os.path.exists(null_distrib_path):
            print(f"Computing null distributions for {model.name} ...")
        null_distribs = compute_null_distribution(
            model_rdms_path=rdm_path,
            brain_rdms_path=roi_path,
            null_distrib_path=null_distrib_path,
            skips=skips,
            permutations_idx=permutations_idx,
        )
        for key, null_distrib in null_distribs.items():
            layer, roi = key.split("_fmri_")
            [observed_statistic] = df.loc[(df["Layer"] == layer) & (df["ROI"] == roi), "R"].values
            pvalue = compute_significance_against_zero(null_distrib, observed_statistic)
            df.loc[(df["Layer"] == layer) & (df["ROI"] == roi), "pvalue"] = pvalue
        all_models_null_distribs[model.name] = null_distribs
        df["Layer_FullNames"] = df["Layer"]
        if "layers" in model and model.layers:
            layer_order = [lay.replace(".", "_") for lay in model.layers]
        else:
            layer_order = [
                lay.replace(".", "_")
                for lay, _ in fx.model.named_modules()
                if lay != "" and lay != "data_preprocessor" and not re.search(r"\d\.", lay)
            ]
            to_remove = set()
            for i in range(len(layer_order) - 1):
                if layer_order[i + 1].startswith(layer_order[i] + "_"):
                    to_remove.add(layer_order[i])
            layer_order = [lay for lay in layer_order if lay not in to_remove]
        if skips:
            layer_order = [lay for lay in layer_order if lay not in skips]
        df["Layer"] = df["Layer"].map({lay: i for i, lay in enumerate(layer_order)})
        df["ModelType"] = model.type
        if "params" in model:
            df["ModelParams"] = model.params
        if "flops" in model:
            df["ModelFlops"] = model.flops
        if "accuracy" in model:
            df["ModelAccuracy"] = model.accuracy
        if "archtype" in model:
            df["ArchType"] = model.archtype
        df.sort_values(by=["ROI", "Layer"], inplace=True)
        df.to_csv(results_csv_path)
        dfs.append(df)

    print("Plotting alignment ...")
    plotter = Plotting(dfs)
    bmd_plot_per_roi_group(
        plotter,
        normUNC=False,
        model_groups=cfg.comparison_variable,
        all_models_null_distribs=all_models_null_distribs if not cfg.other_folders_compare else None,
    )
    bmd_plot_all_layers_line_roi_groups(plotter, model_groups=cfg.comparison_variable, separate=False)


if __name__ == "__main__":
    main()
