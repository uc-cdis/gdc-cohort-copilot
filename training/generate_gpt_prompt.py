import argparse
import os

import pandas as pd
import yaml

DEFAULT_FILTER_COL = "FILTER"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Input data csv")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save prompt template",
    )
    args = parser.parse_args()
    return args


def main(
    *,
    input_csv: str,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    # check to make sure that the required filter column is in the csv
    assert DEFAULT_FILTER_COL in df.columns

    for idx, row in df.iterrows():
        pass

    return


if __name__ == "__main__":
    args = parse_args()
    main(input_csv=args.input_csv, output_dir=args.output_dir)
