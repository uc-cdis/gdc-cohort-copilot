### Dev Notes

* the generate script will resume an interrupted run by using the existing output csv
* either set the env var OPENAI_API_KEY or enter it when prompted by the generate script

```bash
python 1-generate.py \
--input-csv /path/to/test.csv \
--output-csv /path/to/gpt4o_filters.csv \
--field-value-yaml /path/to/fields_short_v2.yaml

python 2-filter-to-query.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--input-csv /path/to/gpt4o_filters.csv \
--input-col generations \
--output-csv /path/to/generated_queries.csv
```
