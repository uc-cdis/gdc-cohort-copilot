import argparse
import re

import pandas as pd
import yaml

from schema import GDCCohortSchema  # isort: skip

PAT = re.compile(r'"field": "(.*?)"')


def is_valid(x):
    try:
        GDCCohortSchema.model_validate_json(x)
    except:
        return False
    return True


def is_core(x, fields):
    matches = PAT.findall(x)
    return len(set(matches) - fields) == 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv", required=True)
    parser.add_argument("--core-fields-yaml", required=True)
    parser.add_argument("--output-tsv", required=True)
    return parser.parse_args()


def main(args):
    src = pd.read_csv(args.input_tsv, sep="\t")
    with open(args.core_fields_yaml, "r") as f:
        fields = set(yaml.safe_load(f)["keys"])

    empty_mask = src["filters"] == "{}"
    dupe_mask = src["filters"].duplicated()
    valid_mask = src["filters"].apply(is_valid)
    core_mask = src["filters"].apply(lambda x: is_core(x, fields))

    df = src[~empty_mask & ~dupe_mask & valid_mask & core_mask].reset_index(drop=True)
    df["filters"] = df["filters"].str.replace(', "isLoggedIn": false', "")
    df.to_csv(args.output_tsv, sep="\t", index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
