# Data Preprocessing

Our data preprocessing pipeline is straightforward. From a dataset of real, user-generated cohort filters (made using the GDC Cohort Builder), we remove empty filters, duplicate filters, filters with fields outside of our core set of filter properties, and filters which do not validate against the current filter structure model. Since the dataset only contains cohort filter JSONs, we next use a large language model (LLM) to reverse translate the filter JSON to a natural language query. Finally, we split the real user data into training and evaluation splits, where we require the evaluation samples meet certain criteria:
* tokenized length of concatenated query and filter must fit within context length limits of all model types
* filter must result in non-null cohort (so that case-retrieval based metrics are well defined)

Our `run.sh` executes all our preprocessing steps.
