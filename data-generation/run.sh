#!/bin/bash

DATA_DIR=/opt/gpudata/gdc-cohort-data

set -e  # Exit immediately if a command exits with a non-zero status

echo "Generating 100,000 samples"
python 1-naive-sampler.py \
--target_samples 100_000 \
--output_filename /path/to/naive_samples_100k.tsv

echo "Generating 1,000,000 samples"
python 1-naive-sampler.py \
--target_samples 1_000_000 \
--output_filename /path/to/naive_samples_1M.tsv