#!/bin/bash

DATA_DIR=/opt/gpudata/gdc-cohort-data
RAW_COHORTS=$DATA_DIR/cohorts_2024-10-25.tsv
REPO_ROOT=/opt/gpudata/steven/gdc-cohort-copilot

set -e  # Exit immediately if a command exits with a non-zero status

echo "Cleaning user cohorts"
python 1-clean-user-cohorts.py \
--input-tsv $RAW_COHORTS \
--core-fields-yaml $REPO_ROOT/defines/core_fields.yaml \
--output-tsv $DATA_DIR/selected_cohorts.tsv

echo "Generating natural language queries"
python 2-filter-to-query.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--input-csv $DATA_DIR/selected_cohorts.tsv \
--input-filter-col filters \
--output-csv $DATA_DIR/generated_queries.csv

echo "Splitting dataset"
python 3-split-dataset.py \
--models \
    openai-community/gpt2 \
    facebook/bart-base \
    mistralai/Mistral-7B-Instruct-v0.3 \
--input-csv $DATA_DIR/generated_queries.csv \
--output-train-csv $DATA_DIR/train.csv \
--output-test-csv $DATA_DIR/test.csv
