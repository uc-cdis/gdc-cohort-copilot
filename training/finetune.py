"""
run commands

1. sft

python3 ./method/finetune.py \
--preprocess True \
--model mistralai/Mistral-7B-Instruct-v0.3 \
--objective  sft \
--run_no 2 \
--batch_size 64 \
--train_epochs 1 \
--max_seq_length 1024 \
--input_csv /opt/gpudata/gdc-eval/results/datasets/tokenized_train_data.csv \
--dataset_dir /opt/gpudata/gdc-eval/results/datasets/ \
--dataset_path /opt/gpudata/gdc-eval/results/datasets/gdc_eval_train_tokenized.hf \
--output_dir  /opt/gpudata/gdc-eval/results/models/sft \
--prompt_yaml /opt/gpudata/anirudh/git-repos/gdc-eval/ref-data/prompt.yaml \
--baseline_dir /opt/gpudata/gdc-eval/results/baselines/sft


2. continual

python3 ./method/finetune.py \
--preprocess True \
--model gpt2 \
--objective  continual \
--run_no 2 \
--batch_size 32 \
--train_epochs 10 \
--max_seq_length 1024 \
--input_csv /opt/gpudata/gdc-eval/results/datasets/tokenized_train_data.csv \
--dataset_dir /opt/gpudata/gdc-eval/results/datasets/ \
--dataset_path /opt/gpudata/gdc-eval/results/datasets/gdc_eval_train_tokenized.hf \
--output_dir  /opt/gpudata/gdc-eval/results/models/continual \
--prompt_yaml /opt/gpudata/anirudh/git-repos/gdc-eval/ref-data/prompt.yaml \
--baseline_dir /opt/gpudata/gdc-eval/results/baselines/continual

3. seq2seq
python3 ./method/finetune.py \
--preprocess True \
--model facebook/bart-base \
--objective  seq2seq \
--run_no 2 \
--batch_size 32 \
--input_csv /opt/gpudata/gdc-eval/results/datasets/tokenized_train_data.csv \
--dataset_dir /opt/gpudata/gdc-eval/results/datasets/ \
--dataset_path /opt/gpudata/gdc-eval/results/datasets/gdc_eval_train_tokenized.hf \
--output_dir  /opt/gpudata/gdc-eval/results/models/seq2seq \
--prompt_yaml /opt/gpudata/anirudh/git-repos/gdc-eval/ref-data/prompt.yaml \
--baseline_dir /opt/gpudata/gdc-eval/results/baselines/seq2seq
"""

import argparse
import json
import os

import pandas as pd
import torch
import yaml
from _utils import (
    _clean_rewrites,
    _prepare_hf_training_dataset,
    _prepare_training_dataset_tokenized,
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import GroupKFold
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    DataCollatorForSeq2Seq,
    GPT2LMHeadModel,
    GPT2Model,
    GPT2TokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

# constants
PREPROCESS_DATA = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", type=bool, default=False)
    parser.add_argument(
        "--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3"
    )
    parser.add_argument("--objective", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--train_epochs", type=int, default=3)
    parser.add_argument("--run_no", type=int, required=True)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--input_csv", type=str, required=True, help="")
    parser.add_argument("--dataset_dir", type=str, required=True, help="")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--prompt_yaml", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--baseline_dir", type=str, required=True)
    args = parser.parse_args()
    return args


def finetune(
    *,
    preprocess: bool,
    model: str,
    objective: str,
    batch_size: int,
    train_epochs: int,
    lr: float,
    run_no: int,
    max_seq_length: int,
    input_csv: str,
    dataset_dir: str,
    dataset_path: str,
    prompt_yaml: str,
    output_dir: str,
    baseline_dir: str,
):
    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    # prompts
    with open(prompt_yaml) as f:
        prompt_templates = yaml.safe_load(f)

    if preprocess == True:
        # preprocess dataset
        df = pd.read_csv(input_csv)
        # df = _clean_rewrites(df)
        # datasets = _prepare_hf_training_dataset(df, dataset_dir, prompt_templates)
        datasets = _prepare_training_dataset_tokenized(
            df, dataset_dir, prompt_templates
        )
    elif preprocess == False:
        # load the created dataset
        # dataset = load_from_disk(dataset_path)
        datasets = load_from_disk(dataset_path)

    if objective == "sft":
        # SFT
        for prompt_type, dataset in datasets.items():
            tokenizer = AutoTokenizer.from_pretrained(
                model,
                add_eos_token=True,
                use_fast=True,
            )
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = "left"  # needed for flash attention

            # models and lora definition
            base_model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj",
            ]

            peft_config = LoraConfig(
                r=4,
                lora_alpha=16,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
            )
            lora_model = get_peft_model(base_model, peft_config)
            lora_model.print_trainable_parameters()
            lora_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

            # training setup
            sft_training_args = TrainingArguments(
                output_dir=output_dir,
                do_train=True,
                do_eval=False,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=8,
                learning_rate=lr,
                num_train_epochs=train_epochs,
                lr_scheduler_type="linear",
                bf16=True,
                seed=42,
                torch_compile=False,
            )

            def format_dataset(examples):
                return [f"{examples['prompt']}{examples['completion']}"]

            def combine_prompt(examples):
                return {"text": f"{examples['prompt']}{examples['completion']}"}

            def tokenize_examples(examples):
                return tokenizer(examples["text"], max_length=max_seq_length)

            combined_dataset = dataset.map(combine_prompt)
            tokenized_dataset = combined_dataset.map(
                tokenize_examples,
                batched=True,
            )
            trainer = SFTTrainer(
                model=lora_model,
                tokenizer=tokenizer,
                args=sft_training_args,
                train_dataset=tokenized_dataset,
            )
            trainer.train()
            model_savefile = os.path.join(
                baseline_dir,
                f"_{model}_{objective}_{batch_size}_{lr}_{prompt_type}_run_{str(run_no)}",
            )
            trainer.save_model(model_savefile)
            tokenizer.save_pretrained(model_savefile)

            del tokenizer
            del base_model

    elif objective == "continual":
        # CONTINUED PRE-TRAINING
        for prompt_type, dataset in datasets.items():
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            base_model = GPT2LMHeadModel.from_pretrained("gpt2")

            def tokenize_function(examples):
                inputs = tokenizer(
                    examples["completion"],
                    truncation=True,
                    padding="max_length",
                    max_length=max_seq_length,
                )
                inputs["labels"] = inputs["input_ids"].copy()
                return inputs

            def tokenize_function_2(examples):
                combined = [
                    p + "\n" + c
                    for p, c in zip(examples["prompt"], examples["completion"])
                ]
                inputs = tokenizer(
                    combined,
                    truncation=True,
                    padding="max_length",
                    max_length=max_seq_length,
                )
                inputs["labels"] = inputs["input_ids"].copy()
                return inputs

            tokenized_dataset = dataset.map(
                tokenize_function_2,
                batched=True,  # remove_columns=["groups"]
            )
            # print(tokenized_dataset["train"].select([0]))
            tokenized_dataset = tokenized_dataset.with_format("torch")

            continual_training_args = TrainingArguments(
                output_dir=output_dir,
                do_eval=False,
                save_steps=1000,
                save_total_limit=2,
                logging_steps=100,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=8,
                num_train_epochs=train_epochs,
                learning_rate=lr,
                warmup_steps=500,
                fp16=True,
                seed=42,
            )
            trainer = Trainer(
                model=base_model,
                args=continual_training_args,
                train_dataset=tokenized_dataset,
            )
            trainer.train()
            model_savefile = os.path.join(
                baseline_dir,
                f"_{model}_{objective}_{batch_size}_{lr}_{prompt_type}_run_{str(run_no)}",
            )
            trainer.save_model(model_savefile)
            tokenizer.save_pretrained(model_savefile)

            del tokenizer
            del base_model

    elif objective == "seq2seq":
        # SEQ2SEQ
        for prompt_type, dataset in datasets.items():
            tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

            def tokenize_function(example):
                return tokenizer(
                    example["prompt"],
                    text_target=example["completion"],
                    padding=True,
                    truncation=True,
                    max_length=max_seq_length,
                )

            def preprocess(example):
                inputs = tokenizer(
                    example["prompt"],
                    max_length=max_seq_length,
                    truncation=True,
                    padding="max_length",
                )
                targets = tokenizer(
                    example["completion"],
                    max_length=max_seq_length,
                    truncation=True,
                    padding="max_length",
                )

                inputs["labels"] = targets["input_ids"]
                return inputs

            tokenized_dataset = dataset.map(
                preprocess, batched=True, remove_columns=["prompt", "completion"]
            )
            bart_model = BartForConditionalGeneration.from_pretrained(
                "facebook/bart-base"
            )
            data_collator = DataCollatorForSeq2Seq(tokenizer, model=bart_model)
            seq2seq_training_args = Seq2SeqTrainingArguments(
                output_dir=output_dir,
                do_train=True,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=8,
                learning_rate=lr,
                num_train_epochs=train_epochs,
                lr_scheduler_type="linear",
                bf16=True,
                seed=42,
                predict_with_generate=True,
            )
            trainer = Trainer(
                model=bart_model,
                args=seq2seq_training_args,
                train_dataset=tokenized_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
            trainer.train()
            model_savefile = os.path.join(
                baseline_dir,
                f"_{model}_{objective}_{batch_size}_{lr}_{prompt_type}_run_{str(run_no)}",
            )
            trainer.save_model(model_savefile)
            tokenizer.save_pretrained(model_savefile)
    elif objective == "none":
        pass
    return


if __name__ == "__main__":
    args = parse_args()
    finetune(
        preprocess=args.preprocess,
        model=args.model,
        objective=args.objective,
        batch_size=args.batch_size,
        train_epochs=args.train_epochs,
        lr=args.lr,
        run_no=args.run_no,
        max_seq_length=args.max_seq_length,
        input_csv=args.input_csv,
        dataset_dir=args.dataset_dir,
        dataset_path=args.dataset_path,
        prompt_yaml=args.prompt_yaml,
        output_dir=args.output_dir,
        baseline_dir=args.baseline_dir,
    )
