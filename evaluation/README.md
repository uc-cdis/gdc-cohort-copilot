# Evaluation

To evaluate model-generated cohort filters, we do not directly evaluate the generated filter JSON directly, as multiple filters may result in the same resulting cohort. An example of this equivalence is if a filter selects the TCGA program vs a filter which selects each individual TCGA project. Defining all such similar equivalences is a non-trivial task. Rather, we adopt two sets of metrics: case-retrieval and reverse-query based metrics, both further described below. All metrics we define on a per-sample (filter) level so we can compute statistical measures of paired metrics. We report the average of the metrics across the entire test set.

### Case-Retrieval Metrics
Case-retrieval based metrics execute the generated cohort filter to retrieve the cases selected by the filter. We use the empty set of cases for filters which result in errors using the GDC API. We compare the retrieved cases of the generated filter to the retrieved cases of the ground-truth filter. Cases which overlap between the two we treat as true positives. Cases which only are observed in the ground-truth retrieved set are false negatives, and vice-versa cases which only are observed in the generated retrieved set are false positives. Using these definitions, we compute the true positive rate, the intersection-over-union, and an exact-match metric. The exact-match is a binary indicator for if the two sets precisely match, mathematically FP = FN = 0. For TPR and IoU, we apply paired t-tests. For Exact metric, we apply McNemar's test.

### Reverse-Query Metrics
Reverse-query based metrics compare the **semantics** of the generated filter against that of the ground-truth filter. We apply the same reverse-translation tool from our data-preprocessing to the generated filters. We then compute the BERTScore between the natural language queries derived from the generated and ground-truth filters. We specifically use SciBERT as the embedding model for BERTScore given the biomedical research domain of the queries. We apply paired t-tests over the BERTScore metric.

## Evaluation Pipeline
Our evaluation described above is done in steps. We first execute the filters against the GDC API to retrieve cases. We next reverse-translate the generated filters and then compute our evaluation metrics. We finally use an interactive notebook to compute our aggregate results. Our `run.sh` handles the first three of these steps for each of our experiments. An example set of commands is replicated below:

```bash
python 1-retrieve-cases.py \
--input-csv $DATA_DIR/generations/mistral-lora-test-generations.csv \
--input-query-col queries \
--input-filter-col filters \
--output-pkl $DATA_DIR/retrieved_cases/mistral-lora-test-cases.pkl

python 2-filter-to-query.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--input-csv $DATA_DIR/generations/mistral-lora-test-generations.csv \
--input-filter-col filters \
--output-csv $DATA_DIR/queries_for_predicted_filters/mistral-lora-test-queries.csv

python 3-compute-metrics.py \
--true-pkl $DATA_DIR/retrieved_cases/test.pkl \
--pred-pkl $DATA_DIR/retrieved_cases/mistral-lora-test-cases.pkl \
--pred-reverse-queries-csv $DATA_DIR/queries_for_predicted_filters/mistral-lora-test-queries.csv \
--query-col queries \
--output-csv $DATA_DIR/metrics/mistral-lora-test-metrics.csv
```
