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
import asyncio
import json
import os
import urllib

import aiohttp
import pandas as pd
import requests
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio


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


async def _fetch_api_result(session: aiohttp.ClientSession, url: str) -> dict:
    """Fetch a single API result asynchronously"""
    async with session.get(url) as response:
        return await response.json()


async def _api_metric_evaluate(
    session: aiohttp.ClientSession, gt_string: str, gen_string: str
):
    """Evaluate API metrics asynchronously"""
    gt_url = _encode_and_get_request(gt_string)
    gen_url = _encode_and_get_request(gen_string)

    gt_result, gen_result = await asyncio.gather(
        _fetch_api_result(session, gt_url), _fetch_api_result(session, gen_url)
    )

    if "data" in gen_result.keys():
        if gt_result["data"]["hits"] and gen_result["data"]["hits"]:
            gt_ = set([d["case_id"] for d in gt_result["data"]["hits"]])
            gen_ = set(d["case_id"] for d in gen_result["data"]["hits"])

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


async def process_batch(session: aiohttp.ClientSession, batch_df: pd.DataFrame):
    """Process a batch of row concurrently"""
    tasks = []
    for _, row in batch_df.iterrows():
        task = _api_metric_evaluate(session, row["gt"], row["generations"])
        tasks.append(task)

    results = await tqdm_asyncio.gather(*tasks)
    return results


async def async_main(*, input_csv: str, test_data_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    test_df = pd.read_csv(test_data_path)
    gen_df = pd.read_csv(input_csv)

    gen_df["gt"] = test_df["filters"]
    gen_df = _check_valid_gen(gen_df)
    gen_valid_df = gen_df[gen_df["valid"] == True]

    connector = aiohttp.TCPConnector(limit=30)
    async with aiohttp.ClientSession(connector=connector) as session:
        batch_size = 30
        all_results = []

        for i in range(0, len(gen_valid_df), batch_size):
            batch_df = gen_valid_df.iloc[i : i + batch_size]
            batch_results = await process_batch(session, batch_df)
            all_results.extend(batch_results)

    overlaps, scores, jaccard_indices = zip(*all_results)

    gen_valid_df["overlap"] = overlaps
    gen_valid_df["score"] = scores
    gen_valid_df["jaccard_index"] = jaccard_indices

    result_basename = os.path.basename(input_csv)
    result_filename = f"{'_'.join(result_basename.split('.')[:-1])}_processed_v2.csv"
    gen_valid_df.to_csv(os.path.join(output_dir, result_filename))


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
    asyncio.run(
        async_main(
            input_csv=input_csv, test_data_path=test_data_path, output_dir=output_dir
        )
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        input_csv=args.input_csv,
        test_data_path=args.dataset_path,
        output_dir=args.output_dir,
    )
