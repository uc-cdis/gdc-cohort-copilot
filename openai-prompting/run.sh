#!/bin/bash

DATA_DIR=/opt/gpudata/gdc-cohort-data
REPO_ROOT=/opt/gpudata/steven/gdc-cohort-pilot

set -e  # Exit immediately if a command exits with a non-zero status

echo "Prompting GPT-4o"
python 1-generate.py \
--input-csv $DATA_DIR/test.csv \
--output-csv $DATA_DIR/generations/gpt-4o-2024-08-06-test-generations.csv \
--field-value-yaml $REPO_ROOT/defines/field_value_map.yaml
