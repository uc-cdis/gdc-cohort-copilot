import argparse
import hashlib
import json
import random

import numpy as np
import pandas as pd
import yaml

from schema import GDCCohortSchema  # isort: skip


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--field-value-yaml", required=True)
    parser.add_argument("--output-tsv", required=True)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


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

        content.append(
            {
                "op": op,
                "content": {
                    "field": field,
                    "value": final_val,
                },
            },
        )
    return {
        "op": "and",
        "content": content,
    }


def sample_one_nonzero_chisq(df: float) -> int:
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
    n_samples: int,
    special_ranges: dict[str, tuple[int, int]],
    special_ops: list[str],
    df: float = 6.0,
):
    """
    Keeps drawing sizes and building cohorts until we have exactly
    `n_samples` unique, valid filters.
    """
    seen_hashes = set()
    cohorts = []

    while len(cohorts) < n_samples:
        num_fields = sample_one_nonzero_chisq(df)
        fields = random.sample(list(mappings.keys()), num_fields)

        field_vals = {}
        for field in fields:
            opts = mappings[field]
            if field in special_ranges:
                lo, hi = special_ranges[field]
                val = random.randint(lo, hi - 1)
                op = random.choice(special_ops)
            else:
                op = "in"
                if len(opts) > 1:
                    num_vals = random.randint(1, min(len(opts), 5))
                    val = random.sample(opts, num_vals)
                else:
                    val = opts
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


def main(args):
    if args.seed is not None:
        random.seed(args.seed)

    with open(args.field_value_yaml) as f:
        mappings = yaml.safe_load(f)

    special_ranges = {
        "cases.diagnoses.age_at_diagnosis": (0, 32850),
        "cases.diagnoses.year_of_diagnosis": (1900, 2050),
        "cases.exposures.cigarettes_per_day": (0, 999999),
        "cases.exposures.pack_years_smoked": (0, 999999),
        "cases.exposures.tobacco_smoking_onset_year": (1900, 2050),
    }
    special_ops = [">=", ">", "<", "<="]

    cohorts = generate_unique_cohorts(
        mappings=mappings,
        n_samples=args.n_samples,
        special_ranges=special_ranges,
        special_ops=special_ops,
        df=6.0,
    )

    out = pd.DataFrame({"filters": [json.dumps(c) for c in cohorts]})
    out.to_csv(args.output_tsv, sep="\t", index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
