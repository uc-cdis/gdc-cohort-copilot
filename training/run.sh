#!/bin/bash

DATA_DIR=/opt/gpudata/gdc-cohort-data

set -e  # Exit immediately if a command exits with a non-zero status

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

echo "Training GPT2-100k"
python 1-train.py \
--model openai-community/gpt2 \
--input-csv $DATA_DIR/train_synthetic_users+100k.csv \
--log-dir $DATA_DIR/logs/gpt2-100k \
--output-dir $DATA_DIR/models/gpt2-100k \
--batch-size 32 \
--max-epochs 10 \
--lr 5e-5 \
--max-seq-length 1024

echo "Training GPT2-1M"
python 1-train.py \
--model openai-community/gpt2 \
--input-csv $DATA_DIR/train_synthetic_users+1M.csv \
--log-dir $DATA_DIR/logs/gpt2-1m \
--output-dir $DATA_DIR/models/gpt2-1m \
--batch_size 32 \
--max_epochs 10 \
--lr 5e-5 \
--max_seq_length 1024
