# Synthetic Data Generation

We utilize synthetic data in our development of GDC Cohort LLM. While more advanced potential generation methods exist, we utilize naive random sampling. Specifically, we only consider fields and values defined in our core set of filter properties. We sample cohort filter JSONs and rely on our existing filter-to-query reverse translation utilities to complete our synthetic data generation.

### Naive Sampling

At a high-level, to generate a single synthetic cohort filter:
1. select a random number of fields, `n`, to populate using a chi square distribution with 6 degrees of freedom
1. randomly select `n` fields
1. randomly select values
    1. for numerical fields, randomly select an operator and value within a predefined range
    1. for fields with many possible values, randomly select the number of values `m` to sample (up to 5) and randomly select those `m` values
    1. for fields with singleton values, use the singleton value

To run the random sampling:

```bash
python 1-naive-sampler.py \
--target_samples 100_000 \
--output_filename $DATA_DIR/naive_samples_100k.tsv

python 2-filter-to-query.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--input-csv $DATA_DIR/naive_samples_100k.tsv \
--input-filter-col filters \
--output-csv $DATA_DIR/train_synthetic_100k.csv
```
