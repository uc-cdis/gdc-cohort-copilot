import argparse

import numpy as np
import pandas as pd
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

    masks = dict()
    for model, (tok, cfg) in toks.items():
        max_len = cfg.max_position_embeddings
        tok_ids = tok(samples)
        lens = pd.Series([len(x) for x in tok_ids.input_ids])
        masks[model] = lens <= max_len

    mask = df["filters"] != 0
    for model, m in masks.items():
        mask &= m

    possible_idxs = np.argwhere(mask).squeeze()
    rng = np.random.default_rng(seed=42)
    test_idxs = rng.choice(possible_idxs, size=(args.test_size,), replace=False)

    df_train = df.loc[~pd.Series(np.arange(len(df))).isin(test_idxs)]
    df_test = df.loc[test_idxs].sort_index()

    df_train.to_csv(args.output_train_csv, index=False)
    df_test.to_csv(args.output_test_csv, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
