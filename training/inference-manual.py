# load the saved model and its lora layers and generate on test set
"""
run commands
python3 ./training/inference-manual.py \
--input_json /opt/gpudata/anirudh/git-repos/gdc-cohort-pilot/ref/ref_schema_short_v0.json \
--prompt_yaml /opt/gpudata/anirudh/git-repos/gdc-cohort-pilot/ref/prompt.yaml \
--test_dataset_path /opt/gpudata/anirudh/git-repos/gdc-eval/ref-data/man-gdc-100.csv \
--output_filepath /opt/gpudata/gdc-eval/results/inference/new_cohort_repo_test.csv
"""

import argparse
import json
import os

import pandas as pd
import yaml
from _utils import _prepare_hf_test_dataset
from datasets import load_from_disk
from tqdm import trange
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import GuidedDecodingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_dataset_path", type=str, required=True, help="Path to the test dataset"
    )
    parser.add_argument(
        "--input_json", type=str, required=True, help="Path to the input json"
    )
    parser.add_argument(
        "--prompt_yaml", type=str, required=True, help="Path to the prompt yaml"
    )
    parser.add_argument(
        "--output_filepath", type=str, required=True, help="Path to the output csv"
    )
    args = parser.parse_args()
    return args


def main(
    *,  # enforce kwargs
    dataset_path: str,
    input_json: str,
    prompt_yaml: str,
    output_filepath: str,
):
    # prompts
    with open(prompt_yaml) as f:
        prompt_templates = yaml.safe_load(f)
    # ref schema
    with open(input_json, "r") as f:
        ref_json = json.load(f)
    # test data
    inf_df = pd.read_csv(dataset_path)

    # setup deocding params
    decoding_params = GuidedDecodingParams(json=ref_json)

    # set the inf loop for manually curated data
    DEFAULT_QUERY_COL = "QUERY"
    DEFAULT_GROUND_TRUTH_COL = "FILTER"
    DEFAULT_GENERATED_COL = "GPT-4o-RESPONSE"

    queries = {"raw": [], "raw_to_soft": [], "soft_to_raw": []}
    for idx, row in inf_df.iterrows():
        for p_type, p_format in prompt_templates.items():
            queries[p_type].append(
                p_format["query_format"].format(row[DEFAULT_QUERY_COL])
            )

    models_dict = {
        "soft_to_raw": [
            "/opt/gpudata/gdc-eval/results/baselines/sft/_mistralai/Mistral-7B-Instruct-v0.3_sft_64_5e-05_soft_to_raw_run_2",
            "/opt/gpudata/gdc-eval/results/baselines/continual/_gpt2_continual_32_5e-05_soft_to_raw_run_2",
            "/opt/gpudata/gdc-eval/results/baselines/seq2seq/_facebook/bart-base_seq2seq_32_5e-05_soft_to_raw_run_2",
        ],
    }

    # models_dict = {
    #     "raw": [
    #         "/opt/gpudata/gdc-eval/results/baselines/sft/_mistralai/Mistral-7B-Instruct-v0.3_sft_64_5e-05_raw_run_2",
    #         "/opt/gpudata/gdc-eval/results/baselines/continual/_gpt2_continual_32_5e-05_raw_run_2",
    #         "/opt/gpudata/gdc-eval/results/baselines/seq2seq/_facebook/bart-base_seq2seq_32_5e-05_raw_run_2",
    #     ],
    #     "raw_to_soft": [
    #         "/opt/gpudata/gdc-eval/results/baselines/sft/_mistralai/Mistral-7B-Instruct-v0.3_sft_64_5e-05_raw_to_soft_run_2",
    #         "/opt/gpudata/gdc-eval/results/baselines/continual/_gpt2_continual_32_5e-05_raw_to_soft_run_2",
    #         "/opt/gpudata/gdc-eval/results/baselines/seq2seq/_facebook/bart-base_seq2seq_32_5e-05_raw_to_soft_run_2",
    #     ],
    #     "soft_to_raw": [
    #         "/opt/gpudata/gdc-eval/results/baselines/sft/_mistralai/Mistral-7B-Instruct-v0.3_sft_64_5e-05_soft_to_raw_run_2",
    #         "/opt/gpudata/gdc-eval/results/baselines/continual/_gpt2_continual_32_5e-05_soft_to_raw_run_2",
    #         "/opt/gpudata/gdc-eval/results/baselines/seq2seq/_facebook/bart-base_seq2seq_32_5e-05_soft_to_raw_run_2",
    #     ],
    # }

    # vllm setup with same base model and different LORA heads

    for model_type, models in models_dict.items():
        for model in models:
            if "mistral" in model:
                base_model = "mistralai/Mistral-7B-Instruct-v0.3"
                llm = LLM(model=base_model, enable_lora=True)
                sampling_params = SamplingParams(
                    n=1,
                    temperature=0,
                    seed=42,
                    max_tokens=2048,
                    guided_decoding=decoding_params,
                )
                batch_outputs = llm.generate(
                    queries[model_type],
                    sampling_params,
                    lora_request=LoRARequest("gdc-eval", 1, model),
                )
            elif "gpt2" in model:
                llm = LLM(model)
                sampling_params = SamplingParams(
                    n=1,
                    temperature=0,
                    seed=42,
                    max_tokens=2048,
                    guided_decoding=decoding_params,
                )
                batch_outputs = llm.generate(queries[model_type], sampling_params)
            else:
                llm = LLM(model)
                sampling_params = SamplingParams(
                    n=1,
                    temperature=0,
                    seed=42,
                    max_tokens=2048,
                    guided_decoding=decoding_params,
                )
                batch_outputs = llm.generate(queries[model_type], sampling_params)

            batch_outputs = [o.outputs[0].text for o in batch_outputs]
            RESULT_COL = os.path.basename(model)
            inf_df[RESULT_COL] = batch_outputs
            del llm

    inf_df.to_csv(output_filepath, index=False)

    return


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset_path=args.test_dataset_path,
        input_json=args.input_json,
        output_filepath=args.output_filepath,
        prompt_yaml=args.prompt_yaml,
    )
