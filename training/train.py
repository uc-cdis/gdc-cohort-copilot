import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizerFast,
    DataCollatorForSeq2Seq,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)
from trl import SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--max_epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--run_name", type=str, required=True)
    return parser.parse_args()


def main(args):
    dataset = load_dataset("csv", data_files={"train": args.input_csv})
    dataset = dataset["train"].map(
        lambda example: {
            "prompt": example["queries"].strip(),
            "completion": example["filters"],
        },
        remove_columns=["filters", "queries"],
    )
    dataset = dataset.map(
        lambda example: {"text": f"{example['prompt']}{example['completion']}"}
    )
    dataset = dataset.shuffle(seed=42)
    if args.objective == "sft":
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, add_eos_token=True, use_fast=True
        )
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = "left"  # needed for flash attention
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model,
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
        sft_training_args = TrainingArguments(
            output_dir=args.output_dir,
            do_train=True,
            do_eval=False,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=8,
            learning_rate=args.lr,
            num_train_epochs=args.max_epochs,
            lr_scheduler_type="linear",
            bf16=True,
            seed=42,
            torch_compile=False,
        )

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                max_length=args.max_seq_length,
                truncation=True,
                padding=True,
            )

        tok_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=["prompt", "completion"]
        )
        trainer = SFTTrainer(
            model=lora_model,
            tokenizer=tokenizer,
            args=sft_training_args,
            train_dataset=tok_dataset,
        )
        trainer.train()
        model_savefile = os.path.join(
            args.baseline_dir,
            f"_Mistral-7B-Instruct-v0.3_gdc_cohort_pilot_{args.run_name}",
        )
        trainer.save_model(model_savefile)
        tokenizer.save_pretrained(model_savefile)

        del tokenizer
        del base_model

    elif args.objective == "continual":
        tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = GPT2LMHeadModel.from_pretrained(args.model)

        def tokenize_function(examples):
            combined = [
                p + "\n" + c for p, c in zip(examples["prompt"], examples["completion"])
            ]
            inputs = tokenizer(
                combined,
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
            )
            inputs["labels"] = inputs["input_ids"].copy()
            return inputs

        tok_dataset = dataset.map(
            tokenize_function,
            batched=True,
        ).with_format("torch")
        continual_training_args = TrainingArguments(
            output_dir=args.output_dir,
            do_eval=False,
            save_steps=1000,
            save_total_limit=2,
            logging_steps=100,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=8,
            num_train_epochs=args.max_epochs,
            learning_rate=args.lr,
            warmup_steps=500,
            fp16=True,
            seed=42,
        )
        trainer = Trainer(
            model=base_model,
            args=continual_training_args,
            train_dataset=tok_dataset,
        )
        trainer.train()
        model_savefile = os.path.join(
            args.baseline_dir,
            f"_gpt2_gdc_cohort_pilot_{args.run_name}",
        )
        trainer.save_model(model_savefile)
        tokenizer.save_pretrained(model_savefile)

        del tokenizer
        del base_model

    elif args.objective == "seq2seq":
        tokenizer = BartTokenizerFast.from_pretrained(args.model)
        base_model = BartForConditionalGeneration.from_pretrained(args.model)

        def preprocess(example):
            inputs = tokenizer(
                example["prompt"],
                max_length=args.max_seq_length,
                truncation=True,
                padding="max_length",
            )
            targets = tokenizer(
                example["completion"],
                max_length=args.max_seq_length,
                truncation=True,
                padding="max_length",
            )

            inputs["labels"] = targets["input_ids"]
            return inputs

        tok_dataset = dataset.map(
            preprocess, batched=True, remove_columns=["prompt", "completion"]
        )
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=base_model)
        seq2seq_training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            do_train=True,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=8,
            learning_rate=args.lr,
            num_train_epochs=args.max_epochs,
            lr_scheduler_type="linear",
            bf16=True,
            seed=42,
            predict_with_generate=True,
        )
        trainer = Trainer(
            model=base_model,
            args=seq2seq_training_args,
            train_dataset=tok_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        trainer.train()
        model_savefile = os.path.join(
            args.baseline_dir,
            f"_bart_gdc_cohort_pilot_{args.run_name}",
        )
        trainer.save_model(model_savefile)
        tokenizer.save_pretrained(model_savefile)

        del tokenizer
        del base_model
    return


if __name__ == "__main__":
    args = parse_args()
    main(args)
