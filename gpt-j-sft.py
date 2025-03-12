import os
import torch
import shutil
import argparse

from accelerate import Accelerator
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    TrainingArguments,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorForLanguageModeling
)

from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm
from typing import Any, Dict, List, Literal, Optional

from trl import (
    SFTConfig,
    SFTTrainer,
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer import ConstantLengthDataset
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from peft import AutoPeftModelForCausalLM, LoraConfig

@dataclass
class ScriptArguments:
    """
    Arguments common to all scripts.

    Args:
        dataset_name (`str`):
            Dataset name.
        dataset_config (`str` or `None`, *optional*, defaults to `None`):
            Dataset configuration name. Corresponds to the `name` argument of the [`~datasets.load_dataset`] function.
        dataset_train_split (`str`, *optional*, defaults to `"train"`):
            Dataset split to use for training.
        dataset_test_split (`str`, *optional*, defaults to `"test"`):
            Dataset split to use for evaluation.
        gradient_checkpointing_use_reentrant (`bool`, *optional*, defaults to `False`):
            Whether to apply `use_reentrant` for gradient_checkpointing.
        ignore_bias_buffers (`bool`, *optional*, defaults to `False`):
            Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid scalar
            type, inplace operation. See https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992.
    """

    dataset_name: str
    dataset_config: Optional[str] = None
    dataset_train_split: str = "train"
    dataset_test_split: str = "test"
    gradient_checkpointing_use_reentrant: bool = False
    ignore_bias_buffers: bool = False
    score_dataset: bool = False


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def prepare_sample_text(example):
        return f"Document: {example['document']}\n\nSummary: {example['summary']}"

def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("dpo", help="Run the DPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if model_args.model_name_or_path == "CarperAI/openai_summarize_tldr_sft":
        tokenizer_name = "EleutherAI/gpt-j-6b"
    else:
        tokenizer_name = model_args.model_name_or_path
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        padding="longest",
        truncation=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
    
    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )


    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name)
    train_data = dataset[script_args.dataset_train_split].shuffle(seed=42)
    test_data = dataset[script_args.dataset_test_split]
    chars_per_token = chars_token_ratio(train_data, tokenizer)
    
    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        seq_length=training_args.max_seq_length,
    )
    test_dataset = ConstantLengthDataset(
        tokenizer,
        test_data,
        formatting_func=prepare_sample_text,
        seq_length=training_args.max_seq_length,
    )

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save the final model
    final_checkpoint_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.save_model(final_checkpoint_dir)

    model = AutoPeftModelForCausalLM.from_pretrained(final_checkpoint_dir, device_map="auto", torch_dtype=torch.bfloat16) 
    model = model.merge_and_unload()   

    output_merged_dir = os.path.join(training_args.output_dir, "final_merged_checkpoint")
    model.save_pretrained(output_merged_dir, safe_serialization=True)