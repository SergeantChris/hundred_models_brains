from math import ceil
import warnings

import torch
import torch.nn.functional as F

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
import torchextractor as tx

from net2brain.architectures.netsetbase import NetSetBase

from mmaction.datasets.transforms.loading import DecordInit, SampleFrames, DenseSampleFrames, DecordDecode
from mmaction.datasets.transforms.processing import Resize, CenterCrop, ThreeCrop
from mmaction.datasets.transforms.formatting import FormatShape, PackActionInputs
from mmengine.dataset.base_dataset import Compose as MMECompose
from mmengine.dataset.utils import pseudo_collate


def preprocess_mmaction(
    video_path,
    model_name,
    device,
    clip_len,
    frame_interval,
    resize_size,
    crop_type,
    crop_size,
    dense_sampling=False,
    format_shape="NCTHW",
):
    video = {"filename": video_path, "start_index": 0, "modality": "RGB"}
    video = DecordInit()(video)
    num_clips = ceil(video["total_frames"] / (clip_len * frame_interval))
    transform = MMECompose(
        [
            (
                SampleFrames(
                    clip_len=clip_len,
                    frame_interval=frame_interval,
                    num_clips=num_clips,
                    out_of_bound_opt="repeat_last",
                    test_mode=True,
                )
                if not dense_sampling
                else DenseSampleFrames(
                    clip_len=clip_len, frame_interval=frame_interval, num_clips=num_clips, test_mode=True
                )
            ),
            DecordDecode(),
            Resize(scale=(-1, resize_size)),
            ThreeCrop(crop_size=crop_size) if crop_type == "three_crop" else CenterCrop(crop_size=crop_size),
            FormatShape(input_format=format_shape),
            PackActionInputs(),
        ]
    )
    video = transform(video)
    if format_shape == "NCTHW":
        # separate the clips in order to loop them in the extraction function
        video["inputs"] = (
            video["inputs"].reshape((-1, num_clips) + video["inputs"].shape[1:]).transpose(0, 1).contiguous().float()
        )
    else:
        if clip_len == 1 and frame_interval == 1:
            num_clips = 10 if dense_sampling else 3
            if video["inputs"].size(0) % num_clips != 0:
                pad_size = (num_clips - video["inputs"].size(0) % num_clips) % num_clips
                if pad_size > num_clips / 2:
                    padded_data = video["inputs"][: -(video["inputs"].size(0) % num_clips)]
                else:
                    padded_data = F.pad(video["inputs"], (0, 0, 0, 0, 0, 0, 0, pad_size))
                video["inputs"] = padded_data
            video["inputs"] = (
                video["inputs"]
                .reshape((-1, num_clips) + video["inputs"].shape[1:])
                .transpose(0, 1)
                .contiguous()
                .float()
            )
        else:
            video["inputs"] = video["inputs"].unsqueeze(0).float()
    return video


def extract_mmaction(preprocessed_data, layers_to_extract, model, stage):
    layers = NetSetBase.select_model_layers(None, layers_to_extract, None, model)
    normalizer = model.data_preprocessor
    preprocessed_data = pseudo_collate([preprocessed_data])
    preprocessed_data = normalizer(preprocessed_data)["inputs"].squeeze(0)  # squeeze out the fake batch

    device = preprocessed_data.device
    preprocessed_data = preprocessed_data.cpu()
    n_clips = preprocessed_data.shape[0]
    features_all_clips = {}
    for i in range(n_clips):  # sacrifice speed to avoid increasing batch size
        extractor_model = tx.Extractor(model, layers)
        try:
            out, features = extractor_model(preprocessed_data[i].unsqueeze(0).to(device), stage=stage)
        except RuntimeError:
            # pad the input such as that preprocessed_data[i].shape[0] is divisible by 8
            pad_size = (8 - preprocessed_data[i].size(0) % 8) % 8
            if pad_size > 4:
                padded_data = preprocessed_data[i][: -(preprocessed_data[i].size(0) % 8)]
            else:
                padded_data = F.pad(preprocessed_data[i], (0, 0, 0, 0, 0, 0, 0, pad_size))
            out, features = extractor_model(padded_data.unsqueeze(0).to(device), stage=stage)
        del out
        # in mma slowfast has separate keys for slow and fast, so it doesn't need special handling
        features = generic_cleaner_tuples(features)
        for key in features:
            features[key] = features[key].detach().cpu().mean(0)  # average over n_crops
            value = features[key].unsqueeze(0)  # add batch dimension (needed in next steps)
            if key not in features_all_clips:
                features_all_clips[key] = value
            else:
                features_all_clips[key] = torch.stack([features_all_clips[key], value], dim=1).mean(1)
                # average over n_clips
    return features_all_clips


def fixed_cleaner(features):
    clean_dict = {}
    for A_key, subtuple in features.items():
        if type(subtuple) == list or type(subtuple) == tuple:
            if len(subtuple) >= 2:  # if subdict is a list of two values
                keys = [A_key + "_slow", A_key + "_fast"]
                for counter, key in enumerate(keys):
                    clean_dict.update({key: subtuple[counter].cpu()})
            else:
                [value] = subtuple
                clean_dict.update({A_key: value.cpu()})
        elif subtuple.shape[0] != 1:
            # this I added to cover the edge-case of giving a model a batch, specifically the 2dRN model
            # it is a hack and has no place in the general pipeline
            clean_dict.update({A_key: subtuple.mean(0, keepdim=True).cpu()})
        else:
            clean_dict.update({A_key: subtuple.cpu()})
    return clean_dict


def generic_cleaner_tuples(features):
    clean_dict = {}
    for A_key, subtuple in features.items():
        if isinstance(subtuple, (list, tuple)):
            tensor_elements = [elem for elem in subtuple if torch.is_tensor(elem)]
            if len(tensor_elements) == 1:
                clean_dict[A_key] = tensor_elements[0].cpu()
            else:
                new_names = [A_key + f"_{counter}" for counter in range(len(tensor_elements))]
                for counter, key in enumerate(new_names):
                    clean_dict[key] = tensor_elements[counter].cpu()
        else:
            clean_dict[A_key] = subtuple.cpu()
    return clean_dict
