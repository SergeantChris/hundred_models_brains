#!/bin/bash

output_dir="your/path/to/workspace/code/mmaction2/checkpoints"
mkdir -p "$output_dir"

download_root="https://download.openmmlab.com/mmaction/v1.0/recognition"
script_dir=$(dirname "$(realpath "$0")")
paths_file="$script_dir/mma2_checkpoint_list.txt"

declare -A rename_pairs=(
  ["slowonly"]="slow"
  ["uniformerv1"]="uniformer"
)

while IFS= read -r path; do
  full_url="$download_root/$path"
  dir_name=$(dirname "$path")
  for from in "${!rename_pairs[@]}"; do
    to="${rename_pairs[$from]}"
    if [[ "$dir_name" == *"$from"* ]]; then
      dir_name="${dir_name//$from/$to}"
    fi
  done
  dir_name=$(echo "$dir_name" | cut -d'/' -f1)
  file_dir="$output_dir/$dir_name"
  mkdir -p "$file_dir"
  echo "Downloading from: $full_url to $file_dir"
  wget -P "$file_dir" "$full_url"
done < "$paths_file"

echo "All files downloaded to '$output_dir'."