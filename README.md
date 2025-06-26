[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# One Hundred Neural Networks and Brains Watching Videos: Lessons from Alignment (ICLR 2025)
Official paper code, read paper at https://openreview.net/pdf?id=LM4PYXBId5

## Setup Instructions
1. Create a folder named `workspace` somewhere in your system, with subfolders `code` and `data`.
2. Clone this repository into the `code` folder.
3. Set up the environment
   1. Create a new conda environment and activate it: 
      ```bash
      conda create -n repralign python=3.9
      conda activate repralign
      ```
   2. Run `pip install git+https://github.com/cvai-roig-lab/Net2Brain.git@evaluations_enhancements` to install the 
      Net2Brain package at the required branch.
   3. Run `pip uninstall torch torchvision` to uninstall the default PyTorch and torchvision versions. Then run:
      ```bash
      pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
      pip install torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
      ```
   4. To install the rest of the requirements, run:
      ```bash
      pip install -r pip_requirements.txt
      mim install -r mim_requirements.txt
      ```
4. Prepare the [BMD dataset](https://openneuro.org/datasets/ds005165/versions/1.0.3)
   1. Download the fMRI data in 
      `your/path/to/workspace/data/bmd` by running
       ```bash
        #!/bin/bash
       set -e
       #local absolute path to where you want to download the dataset
       LOCAL_DIR="your/path/to/workspace/data/bmd"
       dataset_path="derivatives/versionB/MNI152"
       #create directory paths that mimic the openneuro dataset structure
       mkdir -p "${LOCAL_DIR}/${dataset_path}"
    
       #download the README file
       aws s3 cp --no-sign-request \
       "s3://openneuro.org/ds005165/${dataset_path}/README.txt" \
       "${LOCAL_DIR}/${dataset_path}/"
    
       for sub in {01..10}; do
           data_dir="${LOCAL_DIR}/${dataset_path}/prepared_allvoxel_pkl/sub-${sub}"
           mkdir -p "${data_dir}"
           aws s3 sync --no-sign-request \
           "s3://openneuro.org/ds005165/${dataset_path}/prepared_allvoxel_pkl/sub-${sub}/" \
           "${data_dir}/"
       done
       ```
      Warning! Requires ~50G of disk space.
   2. Run the scripts in this repo `scripts/single_use/bmd_merge_hemispheres.py` and 
      `scripts/single_use/bmd_merge_streams.py` in this order to merge the hemispheres and the streams of the BMD 
      dataset - but first change all references to `your/path/to/workspace` in the scripts to your actual `workspace` 
      path.  
      To save space, you can delete the old folders after running each merging script.
   3. Download the video stimuli from [here](https://boldmomentsdataset.csail.mit.edu/stimuli_metadata/) - place and 
      extract the zip file in `your/path/to/workspace/data/bmd` (you will be asked for the password provided in the 
      README of the above link when unzipping).
5. Prepare the action recognition models
   1. Clone the [mmaction2](https://github.com/open-mmlab/mmaction2.git) repository into the `code` folder - you will
      need it for the model config files.
   2. To download all mmaction2 model checkpoints, run the script `scripts/single_use/download_mma2_checkpoints.sh` - 
      but first change `your/path/to/workspace` in the output directory to your actual `workspace` path.  
      Warning! Requires ~10G of disk space if all models are downloaded.
   3. To fix an installation issue of mmaction2, copy a directory from the cloned mmaction2 repository to the 
      location of your installation in conda by running:
      ```bash
      cp -r your/path/to/workspace/code/mmaction2/mmaction/models/localizers/drn/ your/path/to/conda/envs/repralign/lib/python3.9/site-packages/mmaction/models/localizers/
      ```
      We are not sure why this module is not properly installed, but this fix should work.

## Running the code
1. Open the configuration file `configs/calc_alignment.yaml` and fill in all empty arguments marked by ???. 
2. Run the script `scripts/calc_alignment.py` with the module flag, i.e. `python -m scripts.calc_alignment`.  
   By default this will run the whole list of 92 models (so all models excluding the 7 other-action-dataset models).
   If you want to run a subset of the models, you can comment out or remove any of the models in the `models` list.
   Plots will be saved as pdf files in the directory you ran the script from.
3. The argument `comparison_variable` only affects the grouping in the final plotting, so there is no need to focus 
   on that for the first time running for a model / subset of models.
   The first time running for each model, its results are saved in a csv file under 
   `your/path/to/workspace/data/bmd/RSA_results`, and are loaded from there for all subsequent runs. Then, you can 
   choose to visualize the results grouped by model type (image-object, image-action, or video-action) by keeping 
   the `comparison_variable` as `"ModelType"`, or by architecture type by changing it to `"ArchType"`, and running 
   the script again.

## How to cite
Please cite this work when using the code or adapting it:
```
@inproceedings{
  sartzetaki2025one,
  title={One Hundred Neural Networks and Brains Watching Videos: Lessons from Alignment},
  author={Christina Sartzetaki and Gemma Roig and Cees G. M. Snoek and Iris Groen},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=LM4PYXBId5}
}
```
