import os
import pickle as pkl
import numpy as np


bmd_path = "your/path/to/workspace/data/bmd"
data_path = os.path.join(
    bmd_path,
    "derivatives/versionB/MNI152/prepared_allvoxel_pkl",
)
merged_data_path = os.path.join(
    bmd_path,
    "derivatives/versionB/MNI152/merged_hemispheres_pkl",
)

for subject in range(10):
    sub_roi_path = os.path.join(data_path, f"sub-{(subject + 1):02d}")
    new_sub_roi_path = os.path.join(merged_data_path, f"sub-{(subject + 1):02d}")
    if not os.path.exists(new_sub_roi_path):
        os.makedirs(new_sub_roi_path)
    files = os.listdir(sub_roi_path)
    l_files = [file for file in files if file.endswith(".pkl") and file.split("-")[2].startswith("l")]
    r_files = [file for file in files if file.endswith(".pkl") and file.split("-")[2].startswith("r")]
    l_files.sort()
    r_files.sort()
    for l_file, r_file in zip(l_files, r_files):
        with open(os.path.join(sub_roi_path, l_file), "rb") as f:
            l_data = pkl.load(f)
        with open(os.path.join(sub_roi_path, r_file), "rb") as f:
            r_data = pkl.load(f)
        tr_k = "train_data_allvoxel"
        te_k = "test_data_allvoxel"
        combined_data = {
            tr_k: np.concatenate((l_data[tr_k], r_data[tr_k]), axis=-1),
            te_k: np.concatenate((l_data[te_k], r_data[te_k]), axis=-1),
        }
        common_name = l_file.split("_")[1].split("-")[1][1:]
        output_file = os.path.join(
            new_sub_roi_path,
            f"sub-{(subject + 1):02d}_roi-{common_name}_betas_normalized.pkl",
        )
        with open(output_file, "wb") as f:
            pkl.dump(combined_data, f)
    for file in files:
        if file.endswith(".pkl") and not file.split("-")[2].startswith("l") and not file.split("-")[2].startswith("r"):
            with open(os.path.join(sub_roi_path, file), "rb") as f:
                data = pkl.load(f)
            output_file = os.path.join(new_sub_roi_path, file)
            with open(output_file, "wb") as f:
                pkl.dump(data, f)
