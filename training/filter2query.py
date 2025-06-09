# run command
"""
python3 ./training/filter2query.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--batch_size 16 \
--filter_csv /opt/gpudata/gdc-eval/results/datasets/naive_sampler_100k_v2.tsv \
--output_dir /opt/gpudata/gdc-eval/results/datasets


# 1 M dataset split runs

# cuda:1

python3 ./training/filter2query.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--batch_size 16 \
--filter_csv /opt/gpudata/gdc-eval/results/datasets/naive_sampler_1M_v1_part1.tsv \
--output_dir /opt/gpudata/gdc-eval/results/datasets

# cuda:2

python3 ./training/filter2query.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--batch_size 16 \
--filter_csv /opt/gpudata/gdc-eval/results/datasets/naive_sampler_1M_v1_part2.tsv \
--output_dir /opt/gpudata/gdc-eval/results/datasets

# cuda:3

python3 ./training/filter2query.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--batch_size 16 \
--filter_csv /opt/gpudata/gdc-eval/results/datasets/naive_sampler_1M_v1_part3.tsv \
--output_dir /opt/gpudata/gdc-eval/results/datasets
"""


import argparse
import json
import os

import pandas as pd
from _utils import _prepare_filter_dataset
from tqdm import trange
from vllm import LLM, SamplingParams

DEFAULT_FILTERS_COL = "filters"
TRAIN_FILTERS_COL = "filters_cleaned"


# Prompt
example_1 = """
Example 1:
Dict :
{'op': 'and',
 'content': [{'op': 'in',
   'content': {'field': 'cases.project.program.name', 'value': ['TARGET']}},
  {'op': 'in',
   'content': {'field': 'cases.project.project_id',
    'value': ['TARGET-ALL-P1',
     'TARGET-ALL-P2',
     'TARGET-ALL-P3',
     'TARGET-AML']}},
  {'op': 'in',
   'content': {'field': 'cases.diagnoses.site_of_resection_or_biopsy',
    'value': ['bone marrow']}},
  {'op': 'in',
   'content': {'field': 'cases.samples.tissue_type', 'value': ['tumor']}},
  {'op': 'in',
   'content': {'field': 'cases.samples.tumor_code',
    'value': ['acute lymphoblastic leukemia (all)']}}]}
Sentence : 
acute lymphoblastic leukemia tumor code for bone marrow tumors that belong to TARGET-ALL-P1, TARGET-ALL-P2, TARGET-ALL-P3, TARGET-AML projects. |<eos>|
"""

example_2 = """
Example 2:
Dict:
{'op': 'and',
 'content': [{'op': 'in',
   'content': {'field': 'cases.project.program.name', 'value': ['CGCI']}},
  {'op': 'in',
   'content': {'field': 'cases.project.project_id', 'value': ['CGCI-BLGSP']}},
  {'op': 'in',
   'content': {'field': 'cases.diagnoses.tissue_or_organ_of_origin',
    'value': ['hematopoietic system, nos']}},
  {'op': 'in',
   'content': {'field': 'cases.samples.preservation_method',
    'value': ['ffpe']}}]}
Sentence:
ffpe samples for hematopoietic system, nos that belong to the CGCI-BLGSP project. |<eos>|
"""

prompt = """
Given the following examples of dict and sentence pairs, generate the sentence that describes a new dict between <<>>.
Use the 'field' and it's corresponding 'value' information to correctly identify the different categories.
Examples:

{}

{}

<<{}>>

Sentence:
"""


def generate_query_statements(
    *,  # enforce kwargs
    model: str,
    batch_size: int,
    filter_csv: str,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.basename(model)
    basename = os.path.basename(filter_csv)
    filename = f"{basename.split('.')[0]}_queries.csv"
    result_csv = os.path.join(output_dir, filename)

    sampling_params = SamplingParams(  # greedy
        n=1,
        temperature=0,
        seed=42,
        max_tokens=4096,
        stop=["|<eos>|"],
    )

    llm = LLM(
        model=model,
        trust_remote_code=True,
        enforce_eager=True,
    )

    filters_df = pd.read_csv(filter_csv, sep="\t")

    dataset_df = _prepare_filter_dataset(filters_df, DEFAULT_FILTERS_COL)

    N = len(dataset_df)
    print(f"Size of the filter dataset : {N}")
    for lo in trange(0, N, batch_size, desc="Batched Generation"):
        hi = lo + batch_size
        if hi > N:
            hi = N

        batch_filters = []
        batch_prompts = []
        for i in range(lo, hi):
            current_filter = dataset_df.iloc[i][TRAIN_FILTERS_COL]
            batch_filters.append(str(current_filter))
            batch_prompts.append(
                prompt.format(example_1, example_2, str(current_filter))
            )

        batch_outputs = llm.generate(batch_prompts, sampling_params)
        batch_outputs = [o.outputs[0].text for o in batch_outputs]

        pd.DataFrame(
            {
                "filters": batch_filters,
                "prompts": batch_prompts,
                "queries": batch_outputs,
            }
        ).to_csv(
            result_csv,
            mode="w" if lo == 0 else "a",
            header=lo == 0,
            index=False,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Local or HuggingFace model path/name"
    )
    parser.add_argument(
        "--batch_size", type=int, required=True, help="Batch size for LLM"
    )
    parser.add_argument(
        "--filter_csv", required=True, help="Path to input csv with filter"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save output generation results",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    generate_query_statements(
        model=args.model,
        batch_size=args.batch_size,
        filter_csv=args.filter_csv,
        output_dir=args.output_dir,
    )
