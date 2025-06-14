import argparse

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
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    return parser.parse_args()


def get_mistral_trainer(args, dataset):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        add_eos_token=True,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "left"  # needed for flash attention

    def tokenize_function(xs):
        # pad to max length in batch, truncate if max length exceeded
        return tokenizer(
            [f"{p}{c}" for p, c in zip(xs["prompt"], xs["completion"])],
            truncation=True,
            padding=True,
            max_length=args.max_seq_length,
            return_tensors="pt",
        )

    tok_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "completion"],
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    peft_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    lora_model = get_peft_model(base_model, peft_config)
    lora_model.print_trainable_parameters()
    lora_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    training_args = TrainingArguments(
        output_dir=args.log_dir,
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
    trainer = SFTTrainer(
        model=lora_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tok_dataset,
    )
    return trainer, tokenizer


def get_gpt2_trainer(args, dataset):
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(xs):
        # pad and truncate to max length
        inputs = tokenizer(
            [f"{p}\n{c}" for p, c in zip(xs["prompt"], xs["completion"])],
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_length,
            return_tensors="pt",
        )
        inputs["labels"] = inputs[
            "input_ids"
        ].clone()  # no copy for torch tensor, switched to clone
        return inputs

    tok_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "completion"],
    )

    model = GPT2LMHeadModel.from_pretrained(args.model)
    training_args = TrainingArguments(
        output_dir=args.log_dir,
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
        model=model,
        args=training_args,
        train_dataset=tok_dataset,
    )
    return trainer, tokenizer


def get_bart_trainer(args, dataset):
    tokenizer = BartTokenizerFast.from_pretrained(args.model)

    def tokenize_function(xs):
        # pad and truncate to max length
        inputs = tokenizer(
            xs["prompt"],
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_length,
            return_tensors="pt",
        )
        # pad and truncate to max length
        targets = tokenizer(
            xs["completion"],
            truncation=True,
            padding="max_length",
            max_length=args.max_seq_length,
            return_tensors="pt",
        )
        inputs["labels"] = targets[
            "input_ids"
        ].clone()  # no copy for torch tensor, switched to clone
        return inputs

    tok_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["prompt", "completion"],
    )
    model = BartForConditionalGeneration.from_pretrained(args.model)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.log_dir,
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
        model=model,
        args=training_args,
        train_dataset=tok_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer, tokenizer


def main(args):
    dataset = load_dataset("csv", data_files={"train": args.input_csv})
    dataset = dataset["train"].map(
        lambda x: {
            "prompt": x["queries"].strip(),
            "completion": x["filters"],
        },
        remove_columns=["filters", "queries"],
    )
    dataset = dataset.shuffle(seed=42)

    if "mistral" in args.model:
        trainer, tokenizer = get_mistral_trainer(args, dataset)
    elif "gpt2" in args.model:
        trainer, tokenizer = get_gpt2_trainer(args, dataset)
    elif "bart" in args.model:
        trainer, tokenizer = get_bart_trainer(args, dataset)
    else:
        raise NotImplementedError(f"Unknown model type: {args.model}")

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
