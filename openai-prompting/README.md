# Comparison to GPT-4o

We compare our open-source, small-scale GDC Cohort LLM to a GPT-4o prompting-based alternative. In this comparison, we prompt `gpt-4o-2024-08-06` to generate GDC cohort filters. While GPT-4o qualitatively and conceptually understands the GDC cohort filter structure, it is not trained for this task explicitly. To reduce hallucinations, we provide GPT-4o with the list of all possible fields and values from our core-set of filter properties. This list consumes approximately 15,000 tokens. At the time of our experiments, with token caching, generating over our 2,000 evaluation samples cost approximately $40.

To run the GPT-4o inference, run:
```bash
python 1-generate.py \
--input-csv $DATA_DIR/test.csv \
--output-csv $DATA_DIR/generations/gpt-4o-2024-08-06-test-generations.csv \
--field-value-yaml $REPO_ROOT/defines/field_value_map.yaml
```

Some notes on our script:
* the generate script will resume an interrupted run by using the existing output csv
* either set the env var OPENAI_API_KEY or enter it when prompted by the generate script
