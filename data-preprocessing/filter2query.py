import argparse

import pandas as pd
from vllm import LLM, SamplingParams

DEFAULT_FILTERS_COL = "filters"


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


def generate_queries(
    *,  # enforce kwargs
    model: str,
    input_tsv: str,
    output_csv: str,
):
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

    dataset_df = pd.read_csv(input_tsv, sep="\t")

    prompts = [
        prompt.format(example_1, example_2, x) for x in dataset_df[DEFAULT_FILTERS_COL]
    ]

    outputs = llm.generate(prompts, sampling_params)
    outputs = [o.outputs[0].text for o in outputs]
    out_df = pd.DataFrame(
        {
            "filters": dataset_df[DEFAULT_FILTERS_COL],
            "prompts": prompts,
            "queries": outputs,
        }
    )
    out_df.to_csv(output_csv, index=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-tsv", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    generate_queries(
        model=args.model,
        input_tsv=args.input_tsv,
        output_csv=args.output_csv,
    )
