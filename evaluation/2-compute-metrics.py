import argparse
import pickle

import evaluate
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--true-pkl", required=True)
    parser.add_argument("--pred-pkl", required=True)
    parser.add_argument("--pred-reverse-queries-csv", required=True)
    parser.add_argument("--query-col", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    return args


def load_pickle(pkl_path: str) -> tuple[list[str], list[str]]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    queries = [x[0] for x in data]
    cases = [x[1] for x in data]
    return queries, cases


def compute_case_metrics(
    *,  # enforce kwargs
    true_cases: list[list[str]],
    pred_cases: list[list[str]],
) -> dict[str, list[float]]:
    tprs = []
    ious = []
    exacts = []
    for true_set, pred_set in zip(
        tqdm(true_cases, desc="Computing Case Metrics"),
        pred_cases,
    ):
        true_set = set(true_set)
        pred_set = set(pred_set)

        p = len(true_set)
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        tpr = tp / p
        iou = tp / (tp + fp + fn)
        exact = int((fp == 0) & (fn == 0))

        tprs.append(tpr)
        ious.append(iou)
        exacts.append(exact)

    return {
        "tpr": tprs,
        "iou": ious,
        "exact": exacts,
    }


def compute_query_metrics(
    *,  # enforce kwargs
    true_queries: list[str],
    rev_queries: list[str],
) -> dict[str, list[float]]:
    model_name = "allenai/scibert_scivocab_uncased"
    tok = AutoTokenizer.from_pretrained(model_name)

    def _truncate(queries: list[str], title: str = "") -> list[str]:
        overflows = [len(x) > 512 for x in tok(queries)["input_ids"]]
        if title != "":
            title = " " + title.strip()
        print(f"Truncating {sum(overflows)}{title} queries for BERTScore")
        input_ids = tok(queries, max_length=512, truncation=True)["input_ids"]
        return tok.batch_decode(input_ids, skip_special_tokens=True)

    print("Computing BERTScore")
    bertscore = evaluate.load("bertscore")
    scores = bertscore.compute(
        predictions=_truncate(rev_queries, title="reverse-prediction"),
        references=_truncate(true_queries, title="ground-truth"),
        model_type=model_name,
    )

    return {
        "bertscore": scores["f1"],
    }


def main(args):
    rev_df = pd.read_csv(args.pred_reverse_queries_csv)
    rev_queries = rev_df[args.query_col].to_list()
    true_queries, true_cases = load_pickle(args.true_pkl)
    _, pred_cases = load_pickle(args.pred_pkl)

    # check no server errors
    assert not any([len(x) > 0 and x[0].startswith("GOT") for x in pred_cases])

    case_metrics = compute_case_metrics(
        true_cases=true_cases,
        pred_cases=pred_cases,
    )

    query_metrics = compute_query_metrics(
        true_queries=true_queries,
        rev_queries=rev_queries,
    )

    metrics = pd.DataFrame(case_metrics | query_metrics)
    metrics.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
