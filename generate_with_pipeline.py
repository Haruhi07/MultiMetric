import torch
from torch.utils.data import DataLoader

import datasets
from datasets import load_dataset

import transformers
from transformers import (
    AutoModelForCausalLM, 
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    GenerationConfig,
    pipeline
)

import os
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from random import sample, seed

@dataclass
class ScriptArgs:
    model_name_or_path: str
    dataset_name_or_path: str
    split: str
    max_doc_length: int
    max_new_tokens: int
    batch_size: int
    decoding_strategy: str
    save_path: str
    tokenizer_name_or_path: str
    num_return_sequences: int = 1


dm_single_close_quote = "\u2019"  # unicode
dm_double_close_quote = "\u201d"
END_TOKENS = [
    ".",
    "!",
    "?",
    "...",
    "'",
    "`",
    '"',
    dm_single_close_quote,
    dm_double_close_quote,
    ")",
]  # acceptable ways to end a sentence

def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line:
        return line
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    # print line[-1]
    return line + "."


def process_dataset(dataset, tokenizer, max_doc_length, doc_name, summary_name):
    if doc_name == "prompt":
        system_prompt = "You are a useful AI assistant that helps people to summarize reddit posts. Think first and then summarize the given post into a single sentence."
    elif doc_name == "document":
        system_prompt = "You are a useful AI assistant that helps people to summarize news articles. Think first and then summarize the given article into a single sentence."
    def add_prompt(example):
        doc = example[doc_name].replace("TL;DR:", "").strip()
        ref = example[summary_name]
        query = f"{system_prompt}\nDocument: {doc}\n<think>\n"
        label = tokenizer(ref)["input_ids"]
        input_ids = tokenizer(query, truncation=True)["input_ids"]
        return {
            "source": doc,
            "query": query,
            "reference": ref,
            "messages": [
                {
                    "role": "user", 
                    "content": system_prompt+query
                },
            ],
            "input_ids": input_ids,
            "labels": label
        }
    dataset = dataset.map(add_prompt, num_proc=4).filter(lambda x: len(x["input_ids"]) <= max_doc_length)
    return dataset


def get_model_name(model_name_or_path):
    return model_name_or_path.split("/")[1]


def get_gen_config(decoding_strategy, tokenizer):
    gen_config = None
    if decoding_strategy == "greedy":
        gen_config = GenerationConfig(
            min_length = 10,
            max_new_tokens = script_args.max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )
        print(f"Using greedy decoding strategy!")
    elif decoding_strategy == "temp5":
        gen_config = GenerationConfig(
            min_length = 10,
            max_new_tokens = script_args.max_new_tokens,
            do_sample=True,
            temperature=5.0,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=script_args.num_return_sequences,
        )
        print(f"Using random sampling with temprature 5 and return {script_args.num_return_sequences} sequences!")
    elif decoding_strategy == "temp1.5":
        gen_config = GenerationConfig(
            min_length = 10,
            max_new_tokens = script_args.max_new_tokens,
            do_sample=True,
            temperature=1.5,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=script_args.num_return_sequences,
        )
        print(f"Using random sampling with temprature 1.5 and return {script_args.num_return_sequences} sequences!")
    elif decoding_strategy == "beam6":
        gen_config = GenerationConfig(
            min_length = 10,
            max_new_tokens = script_args.max_new_tokens,
            early_stopping=True,
            num_beams=6,
            num_beam_groups=2,
            diversity_penalty=0.1,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=script_args.num_return_sequences,
        )
        print(f"Using beam search with num_beam 6 and return {script_args.num_return_sequences} sequences!")
    if gen_config is None:
        raise ValueError("Unspecified decoding strategy!")
    return gen_config

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArgs))
    script_args = parser.parse_args_into_dataclasses()[0]

    seed(42)

    print(f"output to {script_args.save_path}")

    ##########
    # Prepare model and dataset
    ##########
    if "qwen" in script_args.model_name_or_path:
        lm = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path, 
            device_map="auto",
            attn_implementation='flash_attention_2', #open turn this on for deepseek
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    else:
        lm = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path, 
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    lm.eval()
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if "tldr" in script_args.dataset_name_or_path:
        doc_name = "prompt"
        summary_name = "completion"
    elif "xsum" in script_args.dataset_name_or_path:
        doc_name = "document"
        summary_name = "summary"

    dataset = load_dataset(script_args.dataset_name_or_path, split=script_args.split)
    prompted_dataset = process_dataset(dataset, tokenizer, script_args.max_doc_length, doc_name, summary_name)
    if script_args.split == "train":
        prompted_dataset = prompted_dataset.select(sample(range(len(prompted_dataset)), 1000))
    dataset = prompted_dataset.remove_columns(dataset.column_names + ["query"])

    print(f"Loaded {len(dataset)} examples from {script_args.dataset_name_or_path} {script_args.split} split")

    ##########
    # Generate completion
    ##########
    generation_input_name = ["input_ids", "attention_mask"]
    completion = []
    gen_config = get_gen_config(script_args.decoding_strategy, tokenizer)
    gen_config_dict = gen_config.to_dict()

    pipe = pipeline("text-generation", model=lm, tokenizer=tokenizer, batch_size=8, device_map="auto", **gen_config_dict)
    for output in tqdm(pipe(prompted_dataset["messages"])):
        #print(output)
        candidate = output[script_args.num_return_sequences-1]["generated_text"]
        completion.append(candidate[-1]["content"])

    ##########
    # Save completion to completions/{model}/{strategy}.csv
    ##########
    df = pd.DataFrame({
        "query": prompted_dataset["query"],
        "completion": completion,
        "source": prompted_dataset["source"],
        "reference": prompted_dataset["reference"],
        "messages": prompted_dataset["messages"]
    })
    df.to_csv(script_args.save_path)