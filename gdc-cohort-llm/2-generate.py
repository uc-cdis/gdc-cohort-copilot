import argparse

import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.sampling_params import GuidedDecodingParams

from schema import GDCCohortSchema  # isort: skip

JSON_SCHEMA = GDCCohortSchema.model_json_schema()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=False)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    return args


def main(args):
    queries = pd.read_csv(args.input_csv)["queries"].to_list()

    decoding_params = GuidedDecodingParams(json=JSON_SCHEMA)
    sampling_params = SamplingParams(
        n=1,
        temperature=0,
        seed=42,
        max_tokens=1024,
        guided_decoding=decoding_params,
    )

    enable_lora = False
    lora_request = None
    if args.adapter is not None:
        enable_lora = True
        lora_request = LoRARequest(
            lora_name="gdc-cohort-copilot",
            lora_int_id=1,
            lora_path=args.adapter,
        )

    llm = LLM(
        model=args.model,
        enable_lora=enable_lora,
        max_model_len=1024,  # different than max completion tokens in sampling params
    )

    outputs = llm.generate(
        prompts=queries,
        sampling_params=sampling_params,
        lora_request=lora_request,
    )
    generations = [o.outputs[0].text for o in outputs]
    temp = pd.DataFrame(
        {
            "queries": queries,
            "generations": generations,
        }
    )
    temp.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
