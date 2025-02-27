import os
import shutil
import pickle as pkl
import numpy as np


bmd_path = "your/path/to/workspace/data/bmd"
merged_data_path = os.path.join(
    bmd_path,
    "derivatives/versionB/MNI152/merged_hemispheres_pkl",
)
merged_data_path_2 = os.path.join(
    bmd_path,
    "derivatives/versionB/MNI152/merged_hemispheres_streams_pkl",
)

for subject in range(10):
    sub_roi_path = os.path.join(merged_data_path, f"sub-{(subject + 1):02d}")
    new_sub_roi_path = os.path.join(merged_data_path_2, f"sub-{(subject + 1):02d}")
    if not os.path.exists(new_sub_roi_path):
        os.makedirs(new_sub_roi_path)
    for i in [1, 2]:
        with open(os.path.join(sub_roi_path, f"sub-{(subject + 1):02d}_roi-V{i}v_betas_normalized.pkl"), "rb") as f:
            v_data = pkl.load(f)
        with open(os.path.join(sub_roi_path, f"sub-{(subject + 1):02d}_roi-V{i}d_betas_normalized.pkl"), "rb") as f:
            d_data = pkl.load(f)
        tr_k = "train_data_allvoxel"
        te_k = "test_data_allvoxel"
        combined_data = {
            tr_k: np.concatenate((v_data[tr_k], d_data[tr_k]), axis=-1),
            te_k: np.concatenate((v_data[te_k], d_data[te_k]), axis=-1),
        }
        common_name = f"V{i}"
        output_file = os.path.join(
            new_sub_roi_path,
            f"sub-{(subject + 1):02d}_roi-{common_name}_betas_normalized.pkl",
        )
        with open(output_file, "wb") as f:
            pkl.dump(combined_data, f)
    files = os.listdir(sub_roi_path)
    for file in files:
        if (
            file.endswith(".pkl")
            and not file.split("-")[2].startswith("V1")
            and not file.split("-")[2].startswith("V2")
        ):
            shutil.copy(os.path.join(sub_roi_path, file), os.path.join(new_sub_roi_path, file))
