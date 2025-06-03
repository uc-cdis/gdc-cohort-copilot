import os

import pandas as pd
import yaml
from _utils import _clean_rewrites
from tqdm import tqdm
from transformers import AutoTokenizer, BartTokenizerFast, GPT2TokenizerFast


def main(*, input_csv: str, prompt_yaml: str):

    with open(prompt_yaml) as f:
        prompts = yaml.safe_load(f)

    tokenizers = {
        "mistral": AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3"),
        "gpt2": GPT2TokenizerFast.from_pretrained("gpt2"),
        "bart": BartTokenizerFast.from_pretrained("facebook/bart-base"),
    }

    df = pd.read_csv(input_csv)
    df = _clean_rewrites(df)

    all_queries = []
    all_filters = []

    for i, row in df.iterrows():
        query = row["queries"]
        filter_ = row["filters"]

        all_queries.append(query)
        all_filters.append(filter_)

        cleaned_outputs = row["outputs_cleaned"]
        for o in cleaned_outputs:
            all_queries.append(o)
            all_filters.append(filter_)

    f_df = pd.DataFrame({"queries": all_queries, "responses": all_filters})
    f_df.to_csv(
        "/opt/gpudata/gdc-eval/results/datasets/train_data_processed.csv", index=False
    )

    result = {
        "raw": {"mistral": [], "gpt2": [], "bart": []},
        "raw_to_soft": {"mistral": [], "gpt2": [], "bart": []},
        "soft_to_raw": {"mistral": [], "gpt2": [], "bart": []},
    }

    for i, row in tqdm(f_df.iterrows(), total=df.shape[0]):
        query = row["queries"]
        response = row["responses"]

        for tok_name, tok in tokenizers.items():
            final_prompts = [
                prompts["raw"]["query_format"].format(query)
                + prompts["raw"]["response_format"].format(response),
                prompts["raw_to_soft"]["query_format"].format(query)
                + prompts["raw_to_soft"]["response_format"].format(response),
                prompts["soft_to_raw"]["query_format"].format(query)
                + prompts["soft_to_raw"]["response_format"].format(response),
            ]

            tok.pad_token = tok.eos_token
            tok_counts = []
            for i, fp in enumerate(final_prompts):
                tokens = tok(fp, return_tensors="pt")
                tok_counts.append(len(tokens["input_ids"][0]))

            result["raw"][tok_name].append(tok_counts[0])
            result["raw_to_soft"][tok_name].append(tok_counts[1])
            result["soft_to_raw"][tok_name].append(tok_counts[2])

    print(f"tokenization complete, starting df operation")

    for k, vs in result.items():
        for v in vs.keys():
            col_name = f"{k}_{v}"
            f_df[col_name] = vs[v]

    return f_df


if __name__ == "__main__":
    df = main(
        input_csv="/opt/gpudata/gdc-eval/results/datasets/Mistral-7B-Instruct-v0.3_rewrites_v2.csv",
        prompt_yaml="/opt/gpudata/anirudh/git-repos/gdc-eval/ref-data/prompt.yaml",
    )
    df.to_csv("/opt/gpudata/gdc-eval/results/datasets/tokenized_train_data.csv")
