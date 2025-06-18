#!/bin/bash

DATA_DIR=/opt/gpudata/gdc-cohort-data

set -e  # Exit immediately if a command exits with a non-zero status

echo "Retrieving ground truth cases"
python 1-retrieve-cases.py \
--input-csv $DATA_DIR/test.csv \
--input-query-col queries \
--input-filter-col filters \
--output-pkl $DATA_DIR/retrieved_cases/test.pkl

################################################################################

echo "Retrieving Mistral generation cases"
python 1-retrieve-cases.py \
--input-csv $DATA_DIR/generations/mistral-lora-test-generations.csv \
--input-query-col queries \
--input-filter-col generations \
--output-pkl $DATA_DIR/retrieved_cases/mistral-lora-test-cases.pkl

echo "Generating Mistral queries"
python 2-filter-to-query.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--input-csv $DATA_DIR/generations/mistral-lora-test-generations.csv \
--input-col generations \
--output-csv $DATA_DIR/queries_for_predicted_filters/mistral-lora-test-queries.csv

echo "Computing Mistral metrics"
python 3-compute-metrics.py \
--true-pkl $DATA_DIR/retrieved_cases/test.pkl \
--pred-pkl $DATA_DIR/retrieved_cases/mistral-lora-test-cases.pkl \
--pred-reverse-queries-csv $DATA_DIR/queries_for_predicted_filters/mistral-lora-test-queries.csv \
--query-col queries \
--output-csv $DATA_DIR/metrics/mistral-lora-test-metrics.csv

################################################################################

echo "Retrieving GPT2 generation cases"
python 1-retrieve-cases.py \
--input-csv $DATA_DIR/generations/gpt2-test-generations.csv \
--input-query-col queries \
--input-filter-col generations \
--output-pkl $DATA_DIR/retrieved_cases/gpt2-test-cases.pkl

echo "Generating GPT2 queries"
python 2-filter-to-query.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--input-csv $DATA_DIR/generations/gpt2-test-generations.csv \
--input-col generations \
--output-csv $DATA_DIR/queries_for_predicted_filters/gpt2-test-queries.csv

echo "Computing GPT2 metrics"
python 3-compute-metrics.py \
--true-pkl $DATA_DIR/retrieved_cases/test.pkl \
--pred-pkl $DATA_DIR/retrieved_cases/gpt2-test-cases.pkl \
--pred-reverse-queries-csv $DATA_DIR/queries_for_predicted_filters/gpt2-test-queries.csv \
--query-col queries \
--output-csv $DATA_DIR/metrics/gpt2-test-metrics.csv

################################################################################

echo "Retrieving BART generation cases"
python 1-retrieve-cases.py \
--input-csv $DATA_DIR/generations/bart-test-generations.csv \
--input-query-col queries \
--input-filter-col generations \
--output-pkl $DATA_DIR/retrieved_cases/bart-test-cases.pkl

echo "Generating BART queries"
python 2-filter-to-query.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--input-csv $DATA_DIR/generations/bart-test-generations.csv \
--input-col generations \
--output-csv $DATA_DIR/queries_for_predicted_filters/bart-test-queries.csv

echo "Computing BART metrics"
python 3-compute-metrics.py \
--true-pkl $DATA_DIR/retrieved_cases/test.pkl \
--pred-pkl $DATA_DIR/retrieved_cases/bart-test-cases.pkl \
--pred-reverse-queries-csv $DATA_DIR/queries_for_predicted_filters/bart-test-queries.csv \
--query-col queries \
--output-csv $DATA_DIR/metrics/bart-test-metrics.csv

################################################################################

echo "Retrieving GPT2-100k generation cases"
python 1-retrieve-cases.py \
--input-csv $DATA_DIR/generations/gpt2-100k-test-generations.csv \
--input-query-col queries \
--input-filter-col generations \
--output-pkl $DATA_DIR/retrieved_cases/gpt2-100k-test-cases.pkl

echo "Generating GPT2-100k queries"
python 2-filter-to-query.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--input-csv $DATA_DIR/generations/gpt2-100k-test-generations.csv \
--input-col generations \
--output-csv $DATA_DIR/queries_for_predicted_filters/gpt2-100k-test-queries.csv

echo "Computing GPT2-100k metrics"
python 3-compute-metrics.py \
--true-pkl $DATA_DIR/retrieved_cases/test.pkl \
--pred-pkl $DATA_DIR/retrieved_cases/gpt2-100k-test-cases.pkl \
--pred-reverse-queries-csv $DATA_DIR/queries_for_predicted_filters/gpt2-100k-test-queries.csv \
--query-col queries \
--output-csv $DATA_DIR/metrics/gpt2-100k-test-metrics.csv

################################################################################

echo "Retrieving GPT2-1M generation cases"
python 1-retrieve-cases.py \
--input-csv $DATA_DIR/generations/gpt2-1m-test-generations.csv \
--input-query-col queries \
--input-filter-col generations \
--output-pkl $DATA_DIR/retrieved_cases/gpt2-1m-test-cases.pkl

echo "Generating GPT2-1M queries"
python 2-filter-to-query.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--input-csv $DATA_DIR/generations/gpt2-1m-test-generations.csv \
--input-col generations \
--output-csv $DATA_DIR/queries_for_predicted_filters/gpt2-1m-test-queries.csv

echo "Computing GPT2-1M metrics"
python 3-compute-metrics.py \
--true-pkl $DATA_DIR/retrieved_cases/test.pkl \
--pred-pkl $DATA_DIR/retrieved_cases/gpt2-1m-test-cases.pkl \
--pred-reverse-queries-csv $DATA_DIR/queries_for_predicted_filters/gpt2-1m-test-queries.csv \
--query-col queries \
--output-csv $DATA_DIR/metrics/gpt2-1m-test-metrics.csv

################################################################################

echo "Retrieving GPT-4o generation cases"
python 1-retrieve-cases.py \
--input-csv $DATA_DIR/generations/gpt-4o-2024-08-06-test-generations.csv \
--input-query-col queries \
--input-filter-col generations \
--output-pkl $DATA_DIR/retrieved_cases/gpt-4o-2024-08-06-test-cases.pkl

echo "Generating GPT-4o queries"
python 2-filter-to-query.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--input-csv $DATA_DIR/generations/gpt-4o-2024-08-06-test-generations.csv \
--input-col generations \
--output-csv $DATA_DIR/queries_for_predicted_filters/gpt-4o-2024-08-06-test-queries.csv

echo "Computing GPT-4o metrics"
python 3-compute-metrics.py \
--true-pkl $DATA_DIR/retrieved_cases/test.pkl \
--pred-pkl $DATA_DIR/retrieved_cases/gpt-4o-2024-08-06-test-cases.pkl \
--pred-reverse-queries-csv $DATA_DIR/queries_for_predicted_filters/gpt-4o-2024-08-06-test-queries.csv \
--query-col queries \
--output-csv $DATA_DIR/metrics/gpt-4o-2024-08-06-test-metrics.csv
