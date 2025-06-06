import ast
import json
import os
import random
import re
from collections import deque
from typing import Any, Dict, List

import networkx as nx
import numpy as np
import pandas as pd
import sklearn
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import GroupKFold


def _assemble_cohort_json(field_val_dict):
    """
    function assembles a gdc cohort filter give a dictionary of field and value pairs
    """
    content_list = []
    for key, val in field_val_dict.items():
        if isinstance(val[1], list):
            content = {"op": val[0], "content": {"field": key, "value": val[1]}}
        else:
            if isinstance(val[1], int):
                content = {"op": val[0], "content": {"field": key, "value": val[1]}}
            else:
                content = {"op": val[0], "content": {"field": key, "value": [val[1]]}}

        content_list.append(content)
    # put together the cohort
    data = {"op": "and", "content": content_list}
    return data


def _create_context_size_dataset(context_size):
    base_text = "This is test data "
    all_data = []
    n_samples = 100

    for i in range(n_samples):
        reps = int(context_size / 4) + 1
        text = base_text * reps

        sample = {"id": f"sample_{i}_size_{context_size}", "text": text}

        all_data.append(sample)

    train_dataset = Dataset.from_list(all_data)
    dataset_dict = DatasetDict({"train": train_dataset})
    dataset_dict.save_to_disk(
        f"/opt/gpudata/gdc-eval/results/datasets/{context_size}k_size_samples"
    )
    return train_dataset


def _prepare_filter_dataset(df, filter_col):
    df = df[df[filter_col] != "{}"]
    df = df.drop_duplicates(subset=[filter_col])
    # remove examples containing set_id
    df["set_id"] = df[filter_col].str.contains("set_id")
    df = df[df["set_id"] == False]
    # remove examples containing case_id
    df["case_id"] = df[filter_col].str.contains("case_id")
    df = df[df["case_id"] == False]
    # remove examples containing gene_id
    df["gene_id"] = df[filter_col].str.contains("gene_id")
    df = df[df["gene_id"] == False]
    # remove examples containing ssm_id
    df["ssm_id"] = df[filter_col].str.contains("ssm_id")
    df = df[df["ssm_id"] == False]

    def extract_logged(example):
        d = json.loads(example)
        if "isLoggedIn" in d:
            d.pop("isLoggedIn")
        return str(d)

    # remove 'isLoggedIn' key from filter dict
    df["filters_cleaned"] = df["filters"].apply(extract_logged)

    return df


def _prepare_rewrite_dataset(df, tokenizer):
    # select ~2k for test and rest sort into training
    df["queries_cleaned"] = df.apply(lambda r: r["queries"].split("|<eos>|")[0], axis=1)
    df["n_tokens"] = df.apply(
        lambda x: len(tokenizer.tokenize(x["queries_cleaned"])), axis=1
    )
    df = df[df["n_tokens"] < 1024]

    # keep 15k for training, remaining go into test
    train_size = 15000
    N = len(df)
    test_size = N - train_size

    s_df = sklearn.utils.shuffle(df)
    train_df = s_df[:train_size]
    test_df = s_df[train_size:]

    assert len(test_df) == test_size
    train_df.to_csv(
        "/opt/gpudata/gdc-eval/results/datasets/train_data.csv", index=False
    )
    test_df.to_csv("/opt/gpudata/gdc-eval/results/datasets/test_data.csv", index=False)

    return train_df


# def _clean_rewrites(df):
#     """remove extra formatting from the generated rewrites for queries"""
#     cleaned = [
#         d.split("\n")[1:] if d.startswith("\n") else d.split("\n")
#         for d in df["outputs"].tolist()
#     ]
#     all_cleaned = []

#     for rew in cleaned:
#         if rew[0].startswith("\n"):
#             if rew[0].split("\n")[1].startswith("1. "):
#                 rew_ = [r.split(". ", 1)[-1].strip() for r in rew[1:]]
#             elif rew[0].split("\n").startswith("Option"):
#                 rew_ = [r"Option\s*\d+[.:)]?\s*".strip() for r in rew[1:]]
#         else:
#             rew_ = [r.split(". ", 1)[-1].strip() for r in rew]
#         all_cleaned.append(rew_)
#     df["rewrites_cleaned"] = all_cleaned
#     return df


def _check_valid(output):
    output_type = type(output)
    try:
        output_type = ast.literal_eval(output)
    except:
        pass

    if isinstance(output_type, list):
        return True

    return False


def _clean_invalid_string(output):
    output = output.strip("[]").strip().split("\n")
    # print(output)
    f_output = []
    for o in output:
        o = o.strip()
        if len(o) < 5:
            continue
        else:
            if o[0] == "'":
                o = o[1:]
                if o[-1] == ",":
                    if o[-2] == "'":
                        o = o[:-2]
                elif o[-1] == "'":
                    o = o[:-1]
            f_output.append(o)

    return f_output


def _clean_rewrites(df):
    """remove extra formatting from the generated rewrites for queries"""
    # df["output_valid"] = df["outputs"].apply(_check_valid)

    # df_valid = df[df["output_valid"] == True]
    # df_valid["len"] = df_valid["outputs"].apply(lambda x : len(ast.literal_eval(x)))
    # df_valid.drop(df_valid[df_valid["len"] != 4].index, inplace=True)
    # df_valid.drop(["len"], axis=1, inplace=True)
    # df_valid["outputs_cleaned"] = df_valid["outputs"].copy()

    # df_invalid = df[df["output_valid"] == False]
    # df_invalid["outputs_cleaned"] = df_invalid["outputs"].apply(_clean_invalid_string)
    # df_invalid["len"] = df_invalid["outputs_cleaned"].apply(lambda x : len(x))
    # df_invalid.drop(df_invalid[df_invalid["len"] != 4].index, inplace=True)
    # df_invalid.drop(["len"], axis=1, inplace=True)

    # # now merge the two to get the final dataset
    # print(f"length of valid: {len(df_valid)}")
    # print(f"length of invalid: {len(df_invalid)}")
    # final_df = pd.concat([df_valid, df_invalid])

    df["outputs_cleaned"] = df["outputs"].apply(_clean_invalid_string)
    df["len"] = df["outputs_cleaned"].apply(lambda x: len(x))
    df.drop(df[df["len"] != 4].index, inplace=True)
    print(len(df))

    return df


def _transform_examples(example):
    return {
        "prompt": example["queries"],
        "completion": example["responses"],
    }


def _prepare_hf_training_dataset(df, dataset_dir, prompt_templates):
    """hf training dataset"""
    datasets = {}
    assert len(df) <= 15000

    for key, val in prompt_templates.items():
        queries = []
        responses = []
        prompt_name = key
        query_format = val["query_format"]
        response_format = val["response_format"]

        for idx, row in df.iterrows():
            queries.append(query_format.format(row["queries"]))
            responses.append(response_format.format(row["filters"]))

            cleaned_outputs = row["outputs_cleaned"]
            assert len(cleaned_outputs) == 4
            for o in cleaned_outputs:
                queries.append(query_format.format(o))
                responses.append(response_format.format(row["filters"]))

        data_dict = {
            "queries": queries,
            "responses": responses,
        }

        ds = Dataset.from_dict(data_dict).shuffle(seed=42)
        datasets[prompt_name] = ds

    final_data_dict = {}
    for prompt_name, dataset in datasets.items():
        t_dataset = dataset.map(
            _transform_examples, remove_columns=["queries", "responses"]
        )
        final_data_dict[prompt_name] = t_dataset

    final_dataset = DatasetDict(final_data_dict)
    final_dataset.save_to_disk(os.path.join(dataset_dir, f"gdc_eval_all.hf"))

    return final_dataset


def _prepare_training_dataset_tokenized(df, dataset_dir, prompt_templates):
    """prepares the training data after tokenization"""
    datasets = {}
    cols = [
        "raw_mistral",
        "raw_gpt2",
        "raw_bart",
        "raw_to_soft_mistral",
        "raw_to_soft_gpt2",
        "raw_to_soft_bart",
        "soft_to_raw_mistral",
        "soft_to_raw_gpt2",
        "soft_to_raw_bart",
    ]
    df["valid_ok"] = df.apply(
        lambda r: all(
            [
                v <= 1024
                for v in [
                    r.raw_mistral,
                    r.raw_gpt2,
                    r.raw_bart,
                    r.raw_to_soft_mistral,
                    r.raw_to_soft_gpt2,
                    r.raw_to_soft_bart,
                    r.soft_to_raw_mistral,
                    r.soft_to_raw_gpt2,
                    r.soft_to_raw_bart,
                ]
            ]
        ),
        axis=1,
    )
    df = df[df["valid_ok"] == True]

    print(f"Size of the training data: {len(df)}")

    for key, val in prompt_templates.items():
        queries = []
        responses = []
        prompt_name = key
        query_format = val["query_format"]
        response_format = val["response_format"]

        for idx, row in df.iterrows():
            queries.append(query_format.format(row["queries"]))
            responses.append(response_format.format(row["responses"]))

        data_dict = {
            "queries": queries,
            "responses": responses,
        }

        ds = Dataset.from_dict(data_dict).shuffle(seed=42)
        datasets[prompt_name] = ds

    final_data_dict = {}
    for prompt_name, dataset in datasets.items():
        t_dataset = dataset.map(
            _transform_examples, remove_columns=["queries", "responses"]
        )
        final_data_dict[prompt_name] = t_dataset

    final_dataset = DatasetDict(final_data_dict)
    final_dataset.save_to_disk(
        os.path.join(dataset_dir, f"gdc_eval_train_tokenized.hf")
    )
    return final_dataset


def _prepare_hf_test_dataset(df, dataset_dir, prompt_templates):
    """hf test dataset"""

    datasets = {}
    for key, val in prompt_templates.items():
        queries = []
        prompt_name = key
        query_format = val["query_format"]

        for idx, row in df.iterrows():
            queries.append(query_format.format(row["queries_cleaned"]))

        data_dict = {"queries": queries}
        ds = Dataset.from_dict(data_dict)
        datasets[prompt_name] = ds

    final_dataset = DatasetDict(datasets)
    final_dataset.save_to_disk(os.path.join(dataset_dir, f"gdc_eval_test.hf"))

    return final_dataset


def _extract_dict_substrings(s):
    dict_strs = []
    parsed_dicts = []
    valid_dicts = []
    par_stack = deque()
    current_dict = ""

    s = str(s)

    for i, char in enumerate(s):
        if char == "{":
            par_stack.append((i, char))
        elif char == "}":
            if len(par_stack) == 0:
                pass
            else:
                if par_stack[-1][1] == "{":
                    # process immediate current dict candidate
                    prev_open = par_stack.pop()
                    dict_str = s[prev_open[0] : i + 1]
                    dict_strs.append(dict_str)
                    try:
                        parsed_dict = ast.literal_eval(dict_str)
                        parsed_dicts.append(parsed_dict)
                        if isinstance(parsed_dict, dict):
                            if (
                                len(current_dict) == 0
                            ):  # set parsed as current to get 1st valid dict
                                current_dict = str(parsed_dict)
                            else:
                                if current_dict not in str(
                                    parsed_dict
                                ):  # not nested, new different dict
                                    valid_dicts.append(current_dict)
                                    current_dict = str(parsed_dict)
                                else:
                                    if len(current_dict) < len(str(parsed_dict)):
                                        valid_dicts.append(str(parsed_dict))
                                        current_dict = str(parsed_dict)
                                    else:
                                        pass
                    except (SyntaxError, ValueError):
                        pass
        else:
            pass
    return dict_strs, parsed_dicts, set(valid_dicts)


def _add_graph_components(g, content, parent=None):
    if isinstance(content, dict):
        for d_key, d_value in content.items():
            if d_key == "op":
                g.add_node(d_value, label=d_value)
                if parent:
                    g.add_edge(parent, d_value)
                _add_graph_components(g, content=content.get("content"), parent=d_value)
            elif d_key == "field":
                g.add_node(d_value, label=d_value)
                if parent:
                    g.add_edge(parent, d_value)
                parent = d_value
            elif d_key == "value":
                if isinstance(d_value, list):
                    for val in d_value:
                        g.add_node(str(val), label=str(val))
                        g.add_edge(parent, str(val))
                else:
                    g.add_node(str(d_value), label=str(d_value))
                    g.add_edge(parent, str(d_value))
    elif isinstance(content, list):  # all items are the same level
        for item in content:
            _add_graph_components(g, content=item, parent=parent)


def _build_filter_graph(filter_dict):
    g = nx.DiGraph()
    if not isinstance(filter_dict, dict):
        # filter_dict = ast.literal_eval(filter_dict)
        filter_dict = json.loads(filter_dict)

    root_op = filter_dict.get("op", {})
    root_content = filter_dict.get("content", {})

    if root_op:
        g.add_node(root_op, label=root_op)
        _add_graph_components(g, root_content, parent=root_op)

    return g


def _extract_largest_nested_jsons(s):
    s = str(s)
    stack = deque()  # Track the start index of current JSON objects
    json_candidates = []  # Store tuples of (start_index, end_index, json_str)

    for i, char in enumerate(s):
        if char == "{":
            stack.append(i)
        elif char == "}":
            if stack:
                start = stack.pop()
                json_str = s[start : i + 1]
                try:
                    parsed_json = ast.literal_eval(json_str)
                    if isinstance(parsed_json, dict):
                        json_candidates.append((start, i + 1, json_str))
                except (SyntaxError, ValueError):
                    pass

    # Retain only the largest JSON objects
    json_candidates = sorted(json_candidates, key=lambda x: (x[0], -x[1]))
    largest_jsons = []
    for _, _, candidate in json_candidates:
        if not any(candidate in outer for outer in largest_jsons):
            largest_jsons.append(candidate)

    return largest_jsons


def _is_valid_json(l):
    valids = []
    for j in l:
        try:
            j_ = json.loads(j)
            valids.append(j_)
        except json.JSONDecodeError:
            pass
    return valids


def _filter_disconneted_graphs(l):
    for l_ in l:
        for k, v in l_.items():
            if k == "op":
                if not isinstance(v, str):
                    return False
                else:
                    return True
            break


def _compute_graph_size(text):
    g = _build_filter_graph(text.strip())
    e = g.size()
    n = len(list(g.nodes))
    return n, e
