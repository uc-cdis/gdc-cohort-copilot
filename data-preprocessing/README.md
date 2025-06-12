### Dev Notes

```bash
python 1-clean-user-cohorts.py \
--input-tsv /path/to/raw_cohorts.tsv \
--core-fields-yaml /path/to/fields.yaml \
--output-tsv /path/to/selected_cohorts.tsv

python 2-filter-to-query.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--input-csv /path/to/selected_cohorts.tsv \
--input-filter-col filters \
--output-csv /path/to/generated_queries.csv

python 3-split-dataset.py \
--models \
openai-community/gpt2 \
facebook/bart-base \
mistralai/Mistral-7B-Instruct-v0.3 \
--input-csv /path/to/generated_queries.csv \
--output-train-csv /path/to/train.csv \
--output-test-csv /path/to/test.csv

python naive_sampler.py \
--target_samples 100_000 \
--output_filename /path/to/naive_samples_100k.tsv

python naive_sampler.py \
--target_samples 1000_000 \
--output_filename /path/to/naive_samples_1M.tsv
```
