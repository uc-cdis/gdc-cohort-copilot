#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

echo "Starting SFT training..."
python3 ./training/train.py --model mistralai/Mistral-7B-Instruct-v0.3 --input_csv /opt/gpudata/gdc-cohort-data/train.csv --output_dir /opt/gpudata/gdc-cohort-data/training_runs/sft --baseline_dir /opt/gpudata/gdc-cohort-data/baselines/sft --objective sft --batch_size 64 --max_epochs 1 --lr 5e-5 --max_seq_length 1024 --run_name users

echo "Finished SFT training. Starting Continual training..."
python3 ./training/train.py --model gpt2 --input_csv /opt/gpudata/gdc-cohort-data/train.csv --output_dir /opt/gpudata/gdc-cohort-data/training_runs/continual --baseline_dir /opt/gpudata/gdc-cohort-data/baselines/continual --objective continual --batch_size 32 --max_epochs 10 --lr 5e-5 --max_seq_length 1024 --run_name users

echo "Finished Continual training. Starting Seq2Seq training..."
python3 ./training/train.py --model facebook/bart-base --input_csv /opt/gpudata/gdc-cohort-data/train.csv --output_dir /opt/gpudata/gdc-cohort-data/training_runs/seq2seq --baseline_dir /opt/gpudata/gdc-cohort-data/baselines/seq2seq --objective seq2seq --batch_size 32 --max_epochs 3 --lr 5e-5 --max_seq_length 1024 --run_name users

echo "All initial training runs completed."

echo "Starting 100k synthetic training..."
python3 ./training/train.py --model gpt2 --input_csv /opt/gpudata/gdc-cohort-data/train_synthetic_users+100k.csv --output_dir /opt/gpudata/gdc-cohort-data/training_runs/continual --baseline_dir /opt/gpudata/gdc-cohort-data/baselines/continual --objective continual --batch_size 32 --max_epochs 10 --lr 5e-5 --max_seq_length 1024 --run_name 100k

echo "Finished 100k synthetic training. Starting 1M synthetic training.."
python3 ./training/train.py --model gpt2 --input_csv /opt/gpudata/gdc-cohort-data/train_synthetic_users+1M.csv --output_dir /opt/gpudata/gdc-cohort-data/training_runs/continual --baseline_dir /opt/gpudata/gdc-cohort-data/baselines/continual --objective continual --batch_size 32 --max_epochs 10 --lr 5e-5 --max_seq_length 1024 --run_name 1M