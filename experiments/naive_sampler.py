import argparse
import hashlib
import json
import os
import random

import numpy as np
import pandas as pd
import yaml

from schema import GDCCohortSchema  # isort: skip


def make_cohort_json(field_val_dict):
    """
    Build the GDC cohort filter JSON
    """
    content = []
    for field, (op, val) in field_val_dict.items():
        if isinstance(val, list):
            final_val = val
        elif isinstance(val, (int, float)):
            final_val = val
        else:
            final_val = [val]

        content.append({"op": op, "content": {"field": field, "value": final_val}})
    return {"op": "and", "content": content}


def sample_one_nonzero_chisq(df):
    """
    Draw one sample from chi sqaured dist, round to int, and reject zeros—
    repeat until > 0. This should be iid.
    """
    while True:
        s = int(round(np.random.chisquare(df)))
        if s > 0:
            return s


def generate_unique_cohorts(
    mappings: dict,
    target_samples: int,
    special_cases: list[str],
    special_ranges: dict[str, tuple[int, int]],
    special_ops: list[str],
    df: float = 6.0,
):
    """
    Keeps drawing sizes and building cohorts until we have exactly
    `target_samples` unique, valid filters.
    """
    seen_hashes = set()
    cohorts = []

    while len(cohorts) < target_samples:
        size = sample_one_nonzero_chisq(df)
        fields = random.sample(list(mappings.keys()), size)

        field_vals = {}
        for field in fields:
            opts = mappings[field]
            if field in special_cases:
                lo, hi = special_ranges[field]
                val = random.randint(lo, hi - 1)
                op = random.choice(special_ops)
            else:
                if len(opts) > 1:
                    k = random.randint(1, min(len(opts), 5))
                    val = random.sample(opts, k)
                    op = "in"
                else:
                    val = opts
                    op = "in"
            field_vals[field] = (op, val)

        cohort = make_cohort_json(field_vals)

        # validate
        try:
            GDCCohortSchema.model_validate(cohort)
        except Exception:
            continue

        # uniqueness
        digest = hashlib.md5(json.dumps(cohort, sort_keys=True).encode()).hexdigest()
        if digest in seen_hashes:
            continue

        seen_hashes.add(digest)
        cohorts.append(cohort)

    return cohorts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target_samples", type=int, required=True)
    p.add_argument("--output_filename", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    base = "/opt/gpudata/gdc-cohort-data"
    with open(os.path.join(base, "fields_short_v2.yaml")) as f:
        mappings = yaml.safe_load(f)

    special_cases = [
        "diagnoses.age_at_diagnosis",
        "diagnoses.year_of_diagnosis",
        "exposures.cigarettes_per_day",
        "exposures.pack_years_smoked",
        "exposures.tobacco_smoking_onset_year",
    ]
    special_ranges = {
        "diagnoses.age_at_diagnosis": (0, 32850),
        "diagnoses.year_of_diagnosis": (1900, 2050),
        "exposures.cigarettes_per_day": (0, 999999),
        "exposures.pack_years_smoked": (0, 999999),
        "exposures.tobacco_smoking_onset_year": (1900, 2050),
    }
    special_ops = [">=", ">", "<", "<="]

    cohorts = generate_unique_cohorts(
        mappings=mappings,
        target_samples=args.target_samples,
        special_cases=special_cases,
        special_ranges=special_ranges,
        special_ops=special_ops,
        df=6.0,
    )

    pd.DataFrame({"filters": [json.dumps(c) for c in cohorts]}).to_csv(
        args.output_filename, sep="\t", index=False
    )


if __name__ == "__main__":
    main()
