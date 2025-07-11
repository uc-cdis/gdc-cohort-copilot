import argparse
import os
from getpass import getpass

import pandas as pd
from openai import LengthFinishReasonError, OpenAI
from tqdm import tqdm

from schema import GDCCohortSchema  # isort: skip


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--field-value-yaml", required=True)
    args = parser.parse_args()
    return args


def generate_filter(client, field_values, query):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": (
                    "Construct NCI GDC cohort filters based on the input cohort "
                    "description using the given list of possible fields and values."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is the list of possible fields and values:\n\n"
                    f"{field_values}\n\n"
                    f"Use the above properties to construct a NCI GDC cohort filter for the following cohort description:\n"
                    f"{query}"
                ),
            },
        ],
        response_format=GDCCohortSchema,
        n=1,
        temperature=0,
        max_tokens=1024,
        seed=42,
    )
    result = completion.choices[0].message.parsed

    # match format of input query
    ret = result.model_dump_json(indent=0)
    ret = ret.replace("\n", "")
    ret = ret.replace('",', '", ')
    ret = ret.replace("},", "}, ")
    return ret


def main(args):
    queries = pd.read_csv(args.input_csv)["queries"].to_list()
    with open(args.field_value_yaml, "r") as f:
        field_values = f.read()

    api_key = os.environ.get("OPENAI_API_KEY", None)
    if api_key is None:
        api_key = getpass("Input OPENAI_API_KEY: ")
    client = OpenAI(api_key=api_key)

    total = len(queries)
    start_idx = 0
    if os.path.exists(args.output_csv):
        print("Output CSV exists, resuming from previously interrupted run.")
        temp = pd.read_csv(args.output_csv)
        start_idx = len(temp)
        queries = queries[start_idx:]
    header = start_idx == 0

    with tqdm(total=total, initial=start_idx) as pbar:
        for query in queries:
            try:
                generation = generate_filter(
                    client=client,
                    field_values=field_values,
                    query=query,
                )
            except Exception as e:
                if isinstance(e, LengthFinishReasonError):
                    generation = "NOT ENOUGH TOKENS"
                else:
                    raise e
            temp = pd.DataFrame(
                {
                    "queries": [query],
                    "generations": [generation],
                }
            )
            temp.to_csv(args.output_csv, index=False, mode="a", header=header)
            pbar.update(1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
