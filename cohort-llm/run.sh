#!/bin/bash

DATA_DIR=/opt/gpudata/gdc-cohort-data

set -e  # Exit immediately if a command exits with a non-zero status

################################################################################

echo "Training Mistral"
python 1-train.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--input-csv $DATA_DIR/train.csv \
--log-dir $DATA_DIR/logs/mistral \
--output-dir $DATA_DIR/models/mistral \
--batch-size 64 \
--max-epochs 1 \
--lr 5e-5 \
--max-seq-length 1024

echo "Inferencing Mistral"
python 2-generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--adapter $DATA_DIR/models/mistral \
--input-csv $DATA_DIR/test.csv \
--output-csv $DATA_DIR/generations/mistral-lora-test-generations.csv

################################################################################

echo "Training GPT2"
python 1-train.py \
--model openai-community/gpt2 \
--input-csv $DATA_DIR/train.csv \
--log-dir $DATA_DIR/logs/gpt2 \
--output-dir $DATA_DIR/models/gpt2 \
--batch-size 32 \
--max-epochs 10 \
--lr 5e-5 \
--max-seq-length 1024

echo "Inferencing GPT2"
python 2-generate.py \
--model $DATA_DIR/models/gpt2 \
--input-csv $DATA_DIR/test.csv \
--output-csv $DATA_DIR/generations/gpt2-test-generations.csv

################################################################################

echo "Training BART"
python 1-train.py \
--model facebook/bart-base \
--input-csv $DATA_DIR/train.csv \
--log-dir $DATA_DIR/logs/bart \
--output-dir $DATA_DIR/models/bart \
--batch-size 32 \
--max-epochs 3 \
--lr 5e-5 \
--max-seq-length 1024

echo "Inferencing BART"
python 2-generate.py \
--model $DATA_DIR/models/bart \
--input-csv $DATA_DIR/test.csv \
--output-csv $DATA_DIR/generations/bart-test-generations.csv

################################################################################

echo "Training GPT2-100k"
python 1-train.py \
--model openai-community/gpt2 \
--input-csv $DATA_DIR/train_synthetic_100k+users.csv \
--log-dir $DATA_DIR/logs/gpt2-100k \
--output-dir $DATA_DIR/models/gpt2-100k \
--batch-size 32 \
--max-epochs 10 \
--lr 5e-5 \
--max-seq-length 1024

echo "Inferencing GPT2-100k"
python 2-generate.py \
--model $DATA_DIR/models/gpt2-100k \
--input-csv $DATA_DIR/test.csv \
--output-csv $DATA_DIR/generations/gpt2-100k-test-generations.csv

################################################################################

echo "Training GPT2-1M"
python 1-train.py \
--model openai-community/gpt2 \
--input-csv $DATA_DIR/train_synthetic_1M+users.csv \
--log-dir $DATA_DIR/logs/gpt2-1m \
--output-dir $DATA_DIR/models/gpt2-1m \
--batch-size 32 \
--max-epochs 10 \
--lr 5e-5 \
--max-seq-length 1024

echo "Inferencing GPT2-1M"
python 2-generate.py \
--model $DATA_DIR/models/gpt2-1m \
--input-csv $DATA_DIR/test.csv \
--output-csv $DATA_DIR/generations/gpt2-1m-test-generations.csv
