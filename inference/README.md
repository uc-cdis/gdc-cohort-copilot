### Dev Notes

```
# mistral w/ lora
python 1-generate.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--adapter /path/to/trained/adapter \
--input-csv /path/to/test.csv \
--output-csv /path/to/generations.csv

# gpt2 or bart
python 1-generate.py \
--model /path/to/trained/model \
--input-csv /path/to/test.csv \
--output-csv /path/to/generations.csv

python 2-filter-to-query.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--input-csv /path/to/generations.csv \
--input-col generations \
--output-csv /path/to/generated_queries.csv
```