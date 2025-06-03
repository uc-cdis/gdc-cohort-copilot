"""
# sft
python3 ./method/evaluate_api.py \
--input_csv /opt/gpudata/gdc-eval/results/inference/sft/sft_generations_raw_run_3.csv \
--dataset_path /opt/gpudata/gdc-eval/results/datasets/test_data.csv \
--output_dir /opt/gpudata/gdc-eval/results/inference/processed/

python3 ./method/evaluate_api.py \
--input_csv /opt/gpudata/gdc-eval/results/inference/sft/sft_generations_raw_to_soft_run_3.csv \
--dataset_path /opt/gpudata/gdc-eval/results/datasets/test_data.csv \
--output_dir /opt/gpudata/gdc-eval/results/inference/processed/

python3 ./method/evaluate_api.py \
--input_csv /opt/gpudata/gdc-eval/results/inference/sft/sft_generations_soft_to_raw_run_3.csv \
--dataset_path /opt/gpudata/gdc-eval/results/datasets/test_data.csv \
--output_dir /opt/gpudata/gdc-eval/results/inference/processed/


# continual
python3 ./method/evaluate_api.py \
--input_csv /opt/gpudata/gdc-eval/results/inference/continual/continual_generations_raw_run_3.csv \
--dataset_path /opt/gpudata/gdc-eval/results/datasets/test_data.csv \
--output_dir /opt/gpudata/gdc-eval/results/inference/processed/

python3 ./method/evaluate_api.py \
--input_csv /opt/gpudata/gdc-eval/results/inference/continual/continual_generations_raw_to_soft_run_3.csv \
--dataset_path /opt/gpudata/gdc-eval/results/datasets/test_data.csv \
--output_dir /opt/gpudata/gdc-eval/results/inference/processed/

python3 ./method/evaluate_api.py \
--input_csv /opt/gpudata/gdc-eval/results/inference/continual/continual_generations_soft_to_raw_run_3.csv \
--dataset_path /opt/gpudata/gdc-eval/results/datasets/test_data.csv \
--output_dir /opt/gpudata/gdc-eval/results/inference/processed/

# seq2seq
python3 ./method/evaluate_api.py \
--input_csv /opt/gpudata/gdc-eval/results/inference/seq2seq/seq2seq_generations_raw_run_3.csv \
--dataset_path /opt/gpudata/gdc-eval/results/datasets/test_data.csv \
--output_dir /opt/gpudata/gdc-eval/results/inference/processed/

python3 ./method/evaluate_api.py \
--input_csv /opt/gpudata/gdc-eval/results/inference/seq2seq/seq2seq_generations_raw_to_soft_run_3.csv \
--dataset_path /opt/gpudata/gdc-eval/results/datasets/test_data.csv \
--output_dir /opt/gpudata/gdc-eval/results/inference/processed/

python3 ./method/evaluate_api.py \
--input_csv /opt/gpudata/gdc-eval/results/inference/seq2seq/seq2seq_generations_soft_to_raw_run_3.csv \
--dataset_path /opt/gpudata/gdc-eval/results/datasets/test_data.csv \
--output_dir /opt/gpudata/gdc-eval/results/inference/processed/

"""

import argparse
import json
import os
import urllib

import pandas as pd
import requests
from tqdm import tqdm


def _query_builder(endpoint: str, filters: str, fields: str, size: str, fmt: str):
    api_url = "https://api.gdc.cancer.gov/"

    if fmt.lower() == "json":
        request_query = (
            api_url
            + endpoint
            + "?filters="
            + filters
            + "&fields="
            + fields
            + "&size="
            + size
            + "&format="
            + fmt
            + "&pretty=true"
        )
    else:
        request_query = (
            api_url
            + endpoint
            + "?filters="
            + filters
            + "&fields="
            + fields
            + "&size="
            + size
            + "&format="
            + fmt
        )
    return request_query


def _encode_and_get_request(filter_string: str):
    # percent encoding of filters
    json_string = filter_string  # replace one_filter with input filter variable here
    example_filter = urllib.parse.quote(json_string.encode("utf-8"))

    # specify fields to be returned
    example_fields = ",".join(
        [
            "case_id",
        ]
    )

    # build API query: queryBuilder(endpoint, filters, fields, size, frmat)
    # to specify no filters and/or no fields to return, replace variable with ''
    template_request = _query_builder(
        "cases", example_filter, example_fields, "11315", "json"
    )
    return template_request


def _check_valid_gen(df: pd.DataFrame):
    valids = []
    for i, row in df.iterrows():
        try:
            json.loads(row["generations"])
            valids.append(True)
        except Exception as e:
            valids.append(False)

    df["valid"] = valids
    return df


def _api_metric_evaluate(gt_string: str, gen_string: str):
    gt_result = requests.get(_encode_and_get_request(gt_string))
    gen_result = requests.get(_encode_and_get_request(gen_string))

    gt = gt_result.json()
    gen = gen_result.json()

    if "data" in gen.keys():
        if gt["data"]["hits"] and gen["data"]["hits"]:
            gt_ = set([d["case_id"] for d in gt["data"]["hits"]])
            gen_ = set(d["case_id"] for d in gen["data"]["hits"])

            score = gt_ == gen_
            overlap = len(gt_.intersection(gen_)) / min([len(gt_), len(gen_)]) * 100
            ji = len(gt_.intersection(gen_)) / len(gt_.union(gen_))
        else:
            score = False
            overlap = -1
            ji = -1
    else:
        score = False
        overlap = -1
        ji = -1
    return overlap, score, ji


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    return args


def main(*, input_csv: str, test_data_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    test_df = pd.read_csv(test_data_path)
    gen_df = pd.read_csv(input_csv)

    gen_df["gt"] = test_df["filters"]
    gen_df = _check_valid_gen(gen_df)
    gen_valid_df = gen_df[gen_df["valid"] == True]

    overlaps = []
    scores = []
    jaccard_indices = []

    for i, row in tqdm(gen_valid_df.iterrows(), total=len(gen_valid_df)):

        ovr, scr, ji = _api_metric_evaluate(row["gt"], row["generations"])
        overlaps.append(ovr)
        scores.append(scr)
        jaccard_indices.append(ji)

    gen_valid_df["overlap"] = overlaps
    gen_valid_df["score"] = scores
    gen_valid_df["jaccard_index"] = jaccard_indices
    result_basename = os.path.basename(input_csv)
    result_filename = f"{'_'.join(result_basename.split('.')[:-1])}_processed_v2.csv"
    gen_valid_df.to_csv(os.path.join(output_dir, result_filename))

    return


if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(
        input_csv=args.input_csv,
        test_data_path=args.dataset_path,
        output_dir=args.output_dir,
    )
