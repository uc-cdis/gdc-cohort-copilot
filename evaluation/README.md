### Dev Notes

```bash
python 1-retrieve-cases.py \
--input-csv /path/to/generated_filters.csv \
--input-query-col queries \
--input-filter-col filters \
--output-pkl /path/to/retrieved_cases.pkl

python 2-compute-metrics.py \
--true-pkl /path/to/true_retrieved_cases.pkl \
--pred-pkl /path/to/pred_retrieved_cases.pkl \
--pred-reverse-queries-csv /path/to/queries_for_predicted_filters.csv \
--query-col queries \
--output-csv /path/to/metrics.csv
```
