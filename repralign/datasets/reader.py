from abc import ABC, abstractmethod
from typing import Optional, Tuple
from functools import partial
import os
import warnings

import numpy as np
import pickle as pkl

from repralign.datasets.constants import *
from repralign.utils.common import raw2rdm


class BaseDatasetReader(ABC):
    def __init__(
        self,
        location: str,
        dataset_path: str,
        dataset_version: Optional[str] = None,
        selection: Optional[str] = None,
        average_external: bool = False,
        transform: Optional[partial] = None,
        return_object: str = "paths",
    ):
        self.location = location
        self.dataset_path = dataset_path
        self.dataset_version = dataset_version
        self.selection = selection
        self.average_external = average_external
        self.transform = transform
        self.rdm_dataset = True
        self.apply_transform_to_model_rdms = False

        if return_object == "paths":
            self.stimuli_path, self.roi_path = self.read_data()
        else:
            raise NotImplementedError('Argument `return_object` can only be the net2brain-expected "paths" for now.')

    @abstractmethod
    def read_data(self) -> Tuple[str, str]:
        pass


class BMDReader(BaseDatasetReader):
    def __init__(
        self,
        location: str,
        dataset_path: str,
        dataset_version: str,
        selection: str,
        average_external: bool,
        transform: Optional[partial],
    ):
        if selection not in ["train", "test", "all"]:
            raise ValueError('For this dataset, `selection` must be either "train", "test", or "all".')
        if selection == "train":
            warnings.warn(
                'There is no clear use-case for "train", because for RSA only "test" should be used, '
                'and for encoding only "all".'
            )
        if transform is not None and transform.func != raw2rdm:
            raise ValueError("For this dataset, only the `raw2rdm` transform is supported for now.")
        if transform is None and average_external:
            raise ValueError(
                "Averaging the external dimension is not possible for raw data, please specify `transform` to create RDMs."
            )
        super().__init__(
            location,
            dataset_path,
            dataset_version=dataset_version,
            selection=selection,
            average_external=average_external,
            transform=transform,
        )

    def read_data(self):
        stimuli_path = os.path.join(self.dataset_path, "stimulus_set/stimuli", self.selection)
        data_path = os.path.join(self.dataset_path, "derivatives", self.dataset_version)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Path {data_path} does not exist - check the `dataset_version` argument.")
        rois = {}
        for subject in range(BMDConstants.N_participants):
            sub_roi_path = os.path.join(data_path, f"sub-{(subject+1):02d}")
            for file in os.listdir(sub_roi_path):
                if file.endswith(".pkl"):
                    roi_name = file.split("_")[1].split("-")[1]
                    with open(os.path.join(sub_roi_path, file), "rb") as f:
                        roi = pkl.load(f)
                    # get roi raw data (average across repetitions)
                    if self.selection == "all":
                        roi_train = roi[f"train_data_allvoxel"].mean(axis=1)
                        roi_test = roi[f"test_data_allvoxel"].mean(axis=1)
                        roi_raw = np.concatenate([roi_train, roi_test], axis=0)
                    else:
                        roi_raw = roi[f"{self.selection}_data_allvoxel"].mean(axis=1)
                    if self.transform is None:
                        # save as npy (needed by net2brain encoding function)
                        if not os.path.exists(os.path.join(sub_roi_path, self.selection)):
                            os.makedirs(os.path.join(sub_roi_path, self.selection))
                        npy_path = os.path.join(sub_roi_path, self.selection, f"fmri_{roi_name}.npy")
                        np.save(npy_path, roi_raw)
                    else:
                        # compute rdm
                        roi_rdm = self.transform(roi_raw)
                        # stack subjects together per roi (needed by net2brain rsa function)
                        if roi_name not in rois:
                            rois[roi_name] = np.expand_dims(roi_rdm, axis=0)
                        else:
                            rois[roi_name] = np.concatenate([rois[roi_name], np.expand_dims(roi_rdm, axis=0)], axis=0)
        if rois:
            roi_path = os.path.join(data_path, "all", self.selection)
            if not os.path.exists(roi_path):
                os.makedirs(roi_path)
            for roi_name, roi in rois.items():
                if self.average_external:
                    roi = np.mean(roi, axis=0, keepdims=True)
                npz_path = os.path.join(roi_path, f"fmri_{roi_name}.npz")
                np.savez(npz_path, roi)
        else:
            roi_path = data_path
            self.rdm_dataset = False
        return stimuli_path, roi_path
