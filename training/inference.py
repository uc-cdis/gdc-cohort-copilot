# load the saved model and its lora layers and generate on test set
"""
run commands

1. sft

python3 ./method/inference.py \
--preprocess True \
--objective sft \
--input_json /opt/gpudata/anirudh/git-repos/gdc-eval/ref-data/ref_schema_short_v0.json \
--prompt_yaml /opt/gpudata/anirudh/git-repos/gdc-eval/ref-data/prompt.yaml \
--test_dataset_path /opt/gpudata/gdc-eval/results/datasets/test_data.csv \
--adapter_config_path /opt/gpudata/gdc-eval/results/baselines/sft/_mistralai/ \
--output_dir /opt/gpudata/gdc-eval/results/inference/


2. continual

python3 ./method/inference.py \
--preprocess True \
--objective continual \
--input_json /opt/gpudata/anirudh/git-repos/gdc-eval/ref-data/ref_schema_short_v0.json \
--prompt_yaml /opt/gpudata/anirudh/git-repos/gdc-eval/ref-data/prompt.yaml \
--test_dataset_path /opt/gpudata/gdc-eval/results/datasets/test_data.csv \
--output_dir /opt/gpudata/gdc-eval/results/inference/

3. seq2seq

python3 ./method/inference.py \
--preprocess True \
--objective seq2seq \
--input_json /opt/gpudata/anirudh/git-repos/gdc-eval/ref-data/ref_schema_short_v0.json \
--prompt_yaml /opt/gpudata/anirudh/git-repos/gdc-eval/ref-data/prompt.yaml \
--test_dataset_path /opt/gpudata/gdc-eval/results/datasets/test_data.csv \
--output_dir /opt/gpudata/gdc-eval/results/inference/
"""


import argparse
import gc
import json
import os

import pandas as pd
import torch
import yaml
from _utils import _prepare_hf_test_dataset
from datasets import load_from_disk
from peft import PeftModel
from tqdm import trange
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import GuidedDecodingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", type=bool, default=False)
    parser.add_argument(
        "--objective", type=str, required=True, help="Modeling objective"
    )
    parser.add_argument(
        "--inf_batch_size",
        type=int,
        default=32,
        help="Batch size to be used for inference",
    )
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
        "--adapter_config_path",
        type=str,
        help="Path to the saved adapater configs and info",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/opt/gpudata/gdc-eval/results/datasets/",
        help="Path to the saved adapater configs and info",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save generations.",
    )
    args = parser.parse_args()
    return args


def main(
    *,  # enforce kwargs
    preprocess: bool,
    objective: str,
    batch_size: int,
    dataset_dir: str,
    dataset_path: str,
    input_json: str,
    prompt_yaml: str,
    adapter_path: str,
    output_dir: str,
):
    inf_dir = os.path.join(output_dir, objective)
    os.makedirs(inf_dir, exist_ok=True)
    # prompts
    with open(prompt_yaml) as f:
        prompt_templates = yaml.safe_load(f)

    if preprocess == True:
        df = pd.read_csv(dataset_path)
        datasets = _prepare_hf_test_dataset(df, dataset_dir, prompt_templates)
    else:
        datasets = load_from_disk(dataset_path)
    #
    with open(input_json, "r") as f:
        ref_json = json.load(f)

    # setup deocding params
    decoding_params = GuidedDecodingParams(json=ref_json)

    # vllm setup with same base model and different LORA heads

    if objective == "sft":
        # supervised fine tuning

        base_model = "mistralai/Mistral-7B-Instruct-v0.3"
        llm = LLM(model=base_model, enable_lora=True)
        sampling_params = SamplingParams(
            n=1,
            temperature=0,
            seed=42,
            max_tokens=2048,
            guided_decoding=decoding_params,
        )
        adapter_models = {
            "raw": "Mistral-7B-Instruct-v0.3_sft_64_5e-05_raw_run_2",
            "raw_to_soft": "Mistral-7B-Instruct-v0.3_sft_64_5e-05_raw_to_soft_run_2",
            "soft_to_raw": "Mistral-7B-Instruct-v0.3_sft_64_5e-05_soft_to_raw_run_2",
        }

        for prompt_type, dataset in datasets.items():
            adapter_model = os.path.join(adapter_path, adapter_models[prompt_type])
            result_csv = os.path.join(
                inf_dir, f"{objective}_generations_{prompt_type}_run_3.csv"
            )

            N = len(dataset)
            for lo in trange(0, N, batch_size, desc="Batched inference"):
                hi = lo + batch_size
                if hi > N:
                    hi = N

                batch_prompts = dataset[lo:hi]["queries"]
                batch_outputs = llm.generate(
                    batch_prompts,
                    sampling_params,
                    lora_request=LoRARequest("gdc-eval", 1, adapter_model),
                )
                batch_outputs = [o.outputs[0].text for o in batch_outputs]

                pd.DataFrame(
                    {"queries": batch_prompts, "generations": batch_outputs}
                ).to_csv(
                    result_csv,
                    mode="w" if lo == 0 else "a",
                    header=lo == 0,
                    index=False,
                )
    elif objective == "continual":
        # GPT 2 CONTINUED PRE-TRAINING
        # test_dataset = dataset["test"]
        models = {
            "raw": "_gpt2_continual_32_5e-05_raw_run_2",
            "raw_to_soft": "_gpt2_continual_32_5e-05_raw_to_soft_run_2",
            "soft_to_raw": "_gpt2_continual_32_5e-05_soft_to_raw_run_2",
        }

        sampling_params = SamplingParams(
            n=1,
            temperature=0,
            seed=42,
            max_tokens=512,
            guided_decoding=decoding_params,
        )

        for prompt_type, dataset in datasets.items():
            # load model based on prompt type
            model_path = os.path.join(
                "/opt/gpudata/gdc-eval/results/baselines/continual", models[prompt_type]
            )
            llm = LLM(model_path)

            result_csv = os.path.join(
                inf_dir, f"{objective}_generations_{prompt_type}_run_3.csv"
            )
            N = len(dataset)
            for lo in trange(0, N, batch_size, desc="Batched inference"):
                hi = lo + batch_size
                if hi > N:
                    hi = N

                batch_prompts = dataset[lo:hi]["queries"]

                batch_outputs = llm.generate(
                    batch_prompts,
                    sampling_params,
                )
                batch_outputs = [o.outputs[0].text for o in batch_outputs]

                pd.DataFrame(
                    {"queries": batch_prompts, "generations": batch_outputs}
                ).to_csv(
                    result_csv,
                    mode="w" if lo == 0 else "a",
                    header=lo == 0,
                    index=False,
                )

            del llm
    elif objective == "seq2seq":
        models = {
            "raw": "bart-base_seq2seq_32_5e-05_raw_run_2",
            "raw_to_soft": "bart-base_seq2seq_32_5e-05_raw_to_soft_run_2",
            "soft_to_raw": "bart-base_seq2seq_32_5e-05_soft_to_raw_run_2",
        }

        sampling_params = SamplingParams(
            n=1,
            temperature=0,
            seed=42,
            max_tokens=512,
            guided_decoding=decoding_params,
        )

        for prompt_type, dataset in datasets.items():
            # load model based on prompt type
            model_path = os.path.join(
                "/opt/gpudata/gdc-eval/results/baselines/seq2seq/_facebook/",
                models[prompt_type],
            )
            llm = LLM(model_path)

            result_csv = os.path.join(
                inf_dir, f"{objective}_generations_{prompt_type}_run_3.csv"
            )
            N = len(dataset)
            for lo in trange(0, N, batch_size, desc="Batched inference"):
                hi = lo + batch_size
                if hi > N:
                    hi = N

                batch_prompts = dataset[lo:hi]["queries"]

                batch_outputs = llm.generate(
                    batch_prompts,
                    sampling_params,
                )
                batch_outputs = [o.outputs[0].text for o in batch_outputs]

                pd.DataFrame(
                    {"queries": batch_prompts, "generations": batch_outputs}
                ).to_csv(
                    result_csv,
                    mode="w" if lo == 0 else "a",
                    header=lo == 0,
                    index=False,
                )

            del llm

    return


if __name__ == "__main__":
    args = parse_args()
    main(
        preprocess=args.preprocess,
        objective=args.objective,
        batch_size=args.inf_batch_size,
        dataset_path=args.test_dataset_path,
        input_json=args.input_json,
        adapter_path=args.adapter_config_path,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        prompt_yaml=args.prompt_yaml,
    )
