import argparse

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-train-csv", required=True)
    parser.add_argument("--output-test-csv", required=True)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--models", nargs="+", required=True)
    return parser.parse_args()


def main(args):
    df = pd.read_csv(args.input_csv)[["filters", "queries"]]
    samples = (df["filters"] + df["queries"]).to_list()

    toks = {
        k: (AutoTokenizer.from_pretrained(k), AutoConfig.from_pretrained(k))
        for k in args.models
    }

    # first select possible samples which meet length criteria
    masks = dict()
    for model, (tok, cfg) in toks.items():
        max_len = cfg.max_position_embeddings
        tok_ids = tok(samples)
        lens = pd.Series([len(x) for x in tok_ids.input_ids])
        masks[model] = lens <= max_len

    mask = df["filters"] != 0  # dummy all true mask to start
    for model, m in masks.items():
        mask &= m  # bitwise AND over token length for each model

    possible_idxs = np.argwhere(mask).squeeze()

    # next check that samples do not result in null cohorts
    rng = np.random.default_rng(seed=42)
    rng.shuffle(possible_idxs)
    test_idxs = []
    with tqdm(total=args.test_size, desc="Validating Test Samples") as pbar:
        for idx in possible_idxs:
            response = requests.get(
                "https://api.gdc.cancer.gov/cases",
                params={
                    "filters": df.loc[idx, "filters"],
                    "fields": "submitter_id,case_id",
                    "size": "0",
                },
            )

            # null cohort
            if response.json()["data"]["pagination"]["total"] == 0:
                continue

            test_idxs.append(idx)
            pbar.update(1)
            if len(test_idxs) >= args.test_size:
                break

    if len(test_idxs) < args.test_size:
        print(
            f"Could not select targeted {args.test_size} test samples, "
            f"selection yielded {len(test_idxs)} test samples."
        )
    test_idxs = np.asarray(test_idxs)

    df_train = df.loc[~pd.Series(np.arange(len(df))).isin(test_idxs)]
    df_test = df.loc[test_idxs].sort_index()

    df_train.to_csv(args.output_train_csv, index=False)
    df_test.to_csv(args.output_test_csv, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
