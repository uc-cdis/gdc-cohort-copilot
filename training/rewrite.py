# run command
"""
python3 ./method/rewrite.py --model mistralai/Mistral-7B-Instruct-v0.3 --batch_size 16 --input_csv /opt/gpudata/gdc-eval/results/datasets/Mistral-7B-Instruct-v0.3_generated_queries.csv --output_dir /opt/gpudata/gdc-eval/results/datasets
"""

import argparse
import json
import os

import pandas as pd
from _utils import _prepare_rewrite_dataset
from tqdm import trange
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# prompt = """
# You are an intelligent and effective biomedical language assitant.
# Your task is to restructure and rewrite the given sentence between << >> in simple but equivalent language using relevant medical terminology.
# Remove any redudant terms and generalize where possible.
# Present 4 options in a comma separated list for easy parsing.

# Sentence:
# <<{}>>
# """

prompt = """
[INST]You are an expert biomedical language assistant, and your task is to rewrite a given sentence between <<< >>> using the following rules.
First understand the intent of the sentence and then generate the rewrites.

###
Rules:

1. Use equivalent language and do not deviate from the original intent of the sentence
2. You are allowed to use different equivalent medical terminology where applicable
3. Do not expand any abbreviations or acronyms, use synonyms where possible
4. Check your generations to ensure that they remain in the biomedical domain
5. Strictly follow the output format template provided below. Your output must be a comma separated list, do not generate any other text surrounding it. 
###

Output Template:
[
    'Rewrite 1',
    'Rewrite 2',
    'Rewrite 3',
    'Rewrite 4',
]

Sentence:
<<<
{}
>>>[/INST]
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Local or HuggingFace model path/name"
    )
    parser.add_argument(
        "--batch_size", type=int, required=True, help="Batch size for LLM"
    )
    parser.add_argument(
        "--input_csv", required=True, help="Path to input csv with initial generated"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save output generations results",
    )
    args = parser.parse_args()
    return args


def generate_rewrites(
    *,  # enforce kwargs
    model: str,
    batch_size: int,
    input_csv: str,
    output_dir: str,
):
    model_name = os.path.basename(model)
    filename = f"{model_name}_rewrites_v2.csv"
    result_csv = os.path.join(output_dir, filename)

    sampling_params = SamplingParams(
        n=1,
        temperature=0,
        # use_beam_search=False,
        seed=42,
        max_tokens=4096,
    )

    llm = LLM(
        model=model,
        trust_remote_code=True,
        enforce_eager=True,
    )

    input_df = pd.read_csv(input_csv)
    tokenizer = AutoTokenizer.from_pretrained(model)
    dataset_df = _prepare_rewrite_dataset(input_df, tokenizer)

    N = len(dataset_df)
    print(f"Length of dataset for rewrites: {N}")

    for lo in trange(0, N, batch_size, desc="Batched Generation"):
        hi = lo + batch_size
        if hi > N:
            hi = N

        batch_filters = []
        batch_queries = []
        batch_prompts = []
        for i in range(lo, hi):
            current_filter = dataset_df.iloc[i]["filters"]
            current_query = dataset_df.iloc[i]["queries_cleaned"]
            batch_filters.append(current_filter)
            batch_queries.append(current_query)
            batch_prompts.append(prompt.format(current_query))

        batch_outputs = llm.generate(batch_prompts, sampling_params)
        batch_outputs = [o.outputs[0].text for o in batch_outputs]

        pd.DataFrame(
            {
                "filters": batch_filters,
                "queries": batch_queries,
                "prompts": batch_prompts,
                "outputs": batch_outputs,
            }
        ).to_csv(
            result_csv,
            mode="w" if lo == 0 else "a",
            header=lo == 0,
            index=False,
        )


if __name__ == "__main__":
    args = parse_args()
    generate_rewrites(
        model=args.model,
        batch_size=args.batch_size,
        input_csv=args.input_csv,
        output_dir=args.output_dir,
    )
