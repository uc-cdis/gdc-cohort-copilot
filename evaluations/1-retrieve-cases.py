import argparse
import pickle

import pandas as pd
import requests
from pqdm.threads import pqdm

from schema import GDCCohortSchema  # isort: skip


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--input-query-col", required=True)
    parser.add_argument("--input-filter-col", required=True)
    parser.add_argument("--output-pkl", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    return args


def get_cases(filters: str) -> list[str]:
    response = requests.get(
        "https://api.gdc.cancer.gov/cases",
        params={
            "filters": filters,
            "fields": "case_id",
            "format": "json",
            "size": "46000",  # > num cases in GDC
        },
    )
    data = response.json()["data"]
    cases = [x["case_id"] for x in data["hits"]]
    return cases


def is_valid(filters: str) -> bool:
    try:
        GDCCohortSchema.model_validate_json(filters)
    except Exception as e:
        return False
    return True


def worker(filters: str) -> list[str]:
    if not is_valid(filters):
        return []
    return get_cases(filters)


def main(args):
    df = pd.read_csv(args.input_csv)
    cases = pqdm(df[args.input_filter_col], worker, n_jobs=args.num_workers)
    data = list(zip(df[args.input_query_col], cases))
    with open(args.output_pkl, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
