#!/bin/bash

DATA_DIR=/opt/gpudata/gdc-cohort-data

set -e  # Exit immediately if a command exits with a non-zero status

# echo "Retrieving ground truth cases"
# python 1-retrieve-cases.py \
# --input-csv $DATA_DIR/annotated_test.csv \
# --input-query-col queries \
# --input-filter-col filters \
# --output-pkl $DATA_DIR/retrieved_cases/annotated-test.pkl

################################################################################

# echo "Retrieving GPT2-1M generation cases"
# python 1-retrieve-cases.py \
# --input-csv $DATA_DIR/generations/gpt2-1m-annotated-test-generations.csv \
# --input-query-col queries \
# --input-filter-col generations \
# --output-pkl $DATA_DIR/retrieved_cases/gpt2-1m-annotated-test-cases.pkl

# echo "Generating GPT2-1M queries"
# python 2-filter-to-query.py \
# --model mistralai/Mistral-7B-Instruct-v0.3 \
# --input-csv $DATA_DIR/generations/gpt2-1m-annotated-test-generations.csv \
# --input-col generations \
# --output-csv $DATA_DIR/queries_for_predicted_filters/gpt2-1m-annotated-test-queries.csv

# echo "Computing GPT2-1M metrics"
# python 3-compute-metrics.py \
# --true-pkl $DATA_DIR/retrieved_cases/annotated-test.pkl \
# --pred-pkl $DATA_DIR/retrieved_cases/gpt2-1m-annotated-test-cases.pkl \
# --pred-reverse-queries-csv $DATA_DIR/queries_for_predicted_filters/gpt2-1m-annotated-test-queries.csv \
# --query-col queries \
# --output-csv $DATA_DIR/metrics/gpt2-1m-annotated-test-metrics.csv

################################################################################

echo "Retrieving GPT-4o generation cases"
python 1-retrieve-cases.py \
--input-csv $DATA_DIR/generations/gpt-4o-2024-08-06-annotated-test-generations.csv \
--input-query-col queries \
--input-filter-col generations \
--output-pkl $DATA_DIR/retrieved_cases/gpt-4o-2024-08-06-annotated-test-cases.pkl

echo "Generating GPT-4o queries"
python 2-filter-to-query.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--input-csv $DATA_DIR/generations/gpt-4o-2024-08-06-annotated-test-generations.csv \
--input-col generations \
--output-csv $DATA_DIR/queries_for_predicted_filters/gpt-4o-2024-08-06-annotated-test-queries.csv

echo "Computing GPT-4o metrics"
python 3-compute-metrics.py \
--true-pkl $DATA_DIR/retrieved_cases/annotated-test.pkl \
--pred-pkl $DATA_DIR/retrieved_cases/gpt-4o-2024-08-06-annotated-test-cases.pkl \
--pred-reverse-queries-csv $DATA_DIR/queries_for_predicted_filters/gpt-4o-2024-08-06-annotated-test-queries.csv \
--query-col queries \
--output-csv $DATA_DIR/metrics/gpt-4o-2024-08-06-annotated-test-metrics.csv
