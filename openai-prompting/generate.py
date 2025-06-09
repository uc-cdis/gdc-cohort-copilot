import argparse
import os
from getpass import getpass

import pandas as pd
from openai import OpenAI
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
    ret = result.model_dump_json(indent=0).replace("\n", "").replace('",', '", ')
    return ret


def main(args):
    df = pd.read_csv(args.input_csv)
    with open(args.field_value_yaml, "r") as f:
        field_values = f.read()

    api_key = os.environ.get("OPENAI_API_KEY", None)
    if api_key is None:
        api_key = getpass("Input OPENAI_API_KEY: ")
    client = OpenAI(api_key=api_key)

    generations = []
    for query in tqdm(df["queries"]):
        generation = generate_filter(
            client=client,
            field_values=field_values,
            query=query,
        )
        generations.append(generation)

    df["generations"] = generations
    df[["queries", "generations"]].to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
