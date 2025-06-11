# run command
"""
python3 ./training/rewrite.py \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--batch_size 16 \
--input_csv /opt/gpudata/gdc-cohort-data/train.csv \
--output_dir /opt/gpudata/gdc-cohort-data
"""

import argparse
import json
import os

import pandas as pd
from _utils import _prepare_rewrite_dataset
from tqdm import trange
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

prompt = """
[INST]You are an expert biomedical language assistant, and your task is to rewrite a given sentence between <<< >>> using the following rules.
First understand the intent of the sentence and then generate the rewrites.

###
Rules:

1. Use equivalent language and do not deviate from the original intent of the sentence
2. You are allowed to use different equivalent medical terminology where applicable
3. Do not expand any abbreviations or acronyms, use synonyms where possible
4. Check your generations to ensure that they remain in the biomedical domain
###

Sentence:
<<<
{}
>>>[/INST]
"""

from pydantic import BaseModel


class RewriteOutput(BaseModel):
    rewrite_1: str
    rewrite_2: str
    rewrite_3: str
    rewrite_4: str


JSON_SCHEMA = RewriteOutput.model_json_schema()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    return args


def main(
    *,  # enforce kwargs
    model: str,
    batch_size: int,
    input_csv: str,
    output_dir: str,
):
    model_name = os.path.basename(model)
    filename = f"generated_rewrites.csv"
    result_csv = os.path.join(output_dir, filename)
    decoding_params = GuidedDecodingParams(json=JSON_SCHEMA)
    sampling_params = SamplingParams(
        n=1,
        temperature=0,
        seed=42,
        max_tokens=4096,
        guided_decoding=decoding_params,
    )

    llm = LLM(
        model=model,
        trust_remote_code=True,
        enforce_eager=True,
    )

    input_df = pd.read_csv(input_csv)
    # tokenizer = AutoTokenizer.from_pretrained(model)
    # dataset_df = _prepare_rewrite_dataset(input_df, tokenizer)
    dataset_df = input_df[:32]

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
            current_query = dataset_df.iloc[i]["queries"]
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
    main(
        model=args.model,
        batch_size=args.batch_size,
        input_csv=args.input_csv,
        output_dir=args.output_dir,
    )
