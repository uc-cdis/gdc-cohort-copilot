# imports
import json
import random
import time

import numpy as np
import pandas as pd
import yaml
from _utils import _assemble_cohort_json
from schema import GDCCohortSchema

#
with open(
    "/opt/gpudata/anirudh/git-repos/gdc-eval/ref-data/fields_short_v2.yaml", "r"
) as f:
    cbf_mappings = yaml.safe_load(f)

cbf = list(cbf_mappings.keys())

start = time.time()
# step 1 :
df = 6  # degrees of freedom, also the mean of the chi-squared distribution
num_samples = 1_000_000
samples = np.round(np.random.chisquare(df=df, size=num_samples)).astype(
    int
)  # sample from chi-squared and round to nearest integer

cohorts = []
special_cases = [
    "diagnoses.age_at_diagnosis",
    "diagnoses.year_of_diagnosis",
    "exposures.cigarettes_per_day",
    "exposures.pack_years_smoked",
    "exposures.tobacco_smoking_onset_year",
]
special_field_mappings = {
    "diagnoses.age_at_diagnosis": (0, 32850),
    "diagnoses.year_of_diagnosis": (1900, 2050),
    "exposures.cigarettes_per_day": (0, 999999),
    "exposures.pack_years_smoked": (0, 999999),
    "exposures.tobacco_smoking_onset_year": (1900, 2050),
}
special_field_op_mappings = [">=", ">", "<", "<="]
# step 2 : one cohort per sample
for sample in samples:
    fields = random.sample(list(cbf_mappings.keys()), sample)
    vals_per_field = {}
    # step 3 :
    for field in fields:
        possible_vals = cbf_mappings[field]
        if field in special_cases:
            sp_field_min = special_field_mappings[field][0]
            sp_field_max = special_field_mappings[field][1]
            chosen_values = random.sample(range(sp_field_min, sp_field_max), 1)[0]
            op = random.choice(special_field_op_mappings)
        else:
            # choose how many values and which values
            if len(possible_vals) > 1:
                num_vals = random.sample(range(1, min(len(possible_vals), 5)), 1)[0]
                val_indices = random.sample(range(0, len(possible_vals) - 1), num_vals)
                chosen_values = [possible_vals[idx] for idx in val_indices]
                op = "in"
            else:
                chosen_values = possible_vals
                op = "in"

        vals_per_field[field] = (op, chosen_values)

    # finally put together the cohort json
    cohorts.append(_assemble_cohort_json(vals_per_field))

# validate cohorts to gdc model
valid = []
invalid = []

for cohort in cohorts:
    try:
        GDCCohortSchema.model_validate(cohort)
        valid.append(cohort)
    except Exception as e:
        invalid.append((cohort, e))

print(len(valid))
print(len(invalid))
print(f"computation time: {time.time() - start} secs")

pd.DataFrame({"filters": [json.dumps(c) for c in cohorts]}).to_csv(
    "/opt/gpudata/gdc-eval/results/datasets/naive_sampler_1M_v1.tsv", sep="\t"
)
