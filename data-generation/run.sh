#!/bin/bash

DATA_DIR=/opt/gpudata/gdc-cohort-data

set -e  # Exit immediately if a command exits with a non-zero status

echo "Generating 100,000 samples"
python 1-naive-sampler.py \
--target_samples 100_000 \
--output_filename $DATA_DIR/naive_samples_100k.tsv

echo "Generating natural language queries"
python 2-filter-to-query.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--input-csv $DATA_DIR/naive_samples_100k.tsv \
--input-filter-col filters \
--output-csv $DATA_DIR/train_synthetic_100k.csv

echo "Generating 1,000,000 samples"
python 1-naive-sampler.py \
--target_samples 1_000_000 \
--output_filename $DATA_DIR/naive_samples_1M.tsv

echo "Generating natural language queries"
python 2-filter-to-query.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--input-csv $DATA_DIR/naive_samples_1M.tsv \
--input-filter-col filters \
--output-csv $DATA_DIR/train_synthetic_1M.csv
