# imports
import argparse
import hashlib
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from _utils import _assemble_cohort_json
from schema import GDCCohortSchema
from scipy import stats

"""
python3 ./training/naive_sampler.py --target_samples 100_000 --output_filename /opt/gpudata/gdc-eval/results/datasets/deficit_samples_v1.tsv
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_samples", type=int, required=True)
    parser.add_argument("--output_filename", type=str, required=True)
    return parser.parse_args()


def main(args):
    data_path = "/opt/gpudata/gdc-cohort-data"
    mapping_file = "fields_short_v2.yaml"
    with open(os.path.join(data_path, mapping_file), "r") as f:
        cbf_mappings = yaml.safe_load(f)

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

    cohorts = []
    seen_hashes = set()
    df = 6
    # Calculate probability of getting zero
    zero_prob = stats.chi2.cdf(0.5, df)  # P(X < 0.5) when rounded becomes 0

    # Oversample by safety factor
    safety_factor = 1.2  # 100% extra
    oversample_size = int(args.target_samples / (1 - zero_prob) * safety_factor)
    print(f"oversample size: {oversample_size}")

    samples = np.round(np.random.chisquare(df=df, size=oversample_size)).astype(int)
    nonzero_samples = samples[samples > 0]

    if len(nonzero_samples) < args.target_samples:
        # Fall back to rejection sampling for remaining
        remaining = args.target_samples - len(nonzero_samples)
        additional = args.sample_nonzero_chisquare(df, remaining)
        return np.concatenate([nonzero_samples, additional])

    samples = nonzero_samples[: args.target_samples]
    print(f"sample range min: {min(samples)}")
    print(f"sample range max: {max(samples)}")

    # Assuming you have your samples
    plt.figure(figsize=(10, 6))
    plt.hist(samples, bins=50, alpha=0.7, edgecolor="black")
    plt.title("Histogram of Chi-Square Samples")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig("chi-square hist", dpi=300)

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
                    val_indices = random.sample(range(0, len(possible_vals)), num_vals)
                    chosen_values = [possible_vals[idx] for idx in val_indices]
                    op = "in"
                else:
                    chosen_values = possible_vals
                    op = "in"

            vals_per_field[field] = (op, chosen_values)

        cohort_json = _assemble_cohort_json(vals_per_field)
        # validate the json
        try:
            GDCCohortSchema.model_validate(cohort_json)
        except Exception as e:
            continue
        cohort_str = json.dumps(cohort_json, sort_keys=True)
        cohort_hash = hashlib.md5(cohort_str.encode()).hexdigest()

        if cohort_hash not in seen_hashes:
            seen_hashes.add(cohort_hash)
            cohorts.append(cohort_json)

    pd.DataFrame({"filters": [json.dumps(c) for c in cohorts]}).to_csv(
        args.output_filename, sep="\t", index=False
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
