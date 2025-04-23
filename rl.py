import os
import torch
import shutil
import argparse

from accelerate import Accelerator
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    TrainingArguments,
    AutoTokenizer,
    HfArgumentParser,
)

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from peft import AutoPeftModelForCausalLM, LoraConfig


def get_labeled_dataset(dataset_path):
    score_train_dataset = load_from_disk(f"{dataset_path}/train")#.select(range(100))
    score_test_dataset = load_from_disk(f"{dataset_path}/test")#.select(range(100))
    def set_chosen_and_rejected(data):
        if data["score1"] > data["score2"]:
            chosen = data["completion1"]
            rejected = data["completion2"]
        else:
            chosen = data["completion2"]
            rejected = data["completion1"]
        return {
            "prompt": data["query"],
            "chosen": chosen,
            "rejected": rejected
        }
    
    train_dataset = score_train_dataset.map(
        set_chosen_and_rejected, 
        num_proc=4, 
        remove_columns=score_train_dataset.column_names
    )
    test_dataset = score_test_dataset.map(
        set_chosen_and_rejected, 
        num_proc=4, 
        remove_columns=score_test_dataset.column_names
    )
    return train_dataset, test_dataset

@dataclass
class ScriptArguments:

    dataset_name: str
    dataset_config: Optional[str] = None
    dataset_train_split: str = "train"
    dataset_test_split: str = "test"
    gradient_checkpointing_use_reentrant: bool = False
    ignore_bias_buffers: bool = False
    score_dataset: bool = False


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, DPOConfig, ModelConfig)
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
    shutil.rmtree(training_args.logging_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if model_args.model_name_or_path in ["CarperAI/openai_summarize_tldr_sft", "gpt-j-6b-xsum-sft/final_merged_checkpoint"]:
        tokenizer_name = "EleutherAI/gpt-j-6b"
    else:
        tokenizer_name = model_args.model_name_or_path
    
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=model_args.trust_remote_code,
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
    if script_args.score_dataset:
        train_dataset, test_dataset = get_labeled_dataset(script_args.dataset_name)
    else:
        train_dataset = load_from_disk(f"{script_args.dataset_name}/train")
        test_dataset = load_from_disk(f"{script_args.dataset_name}/test")

    ################
    # Training
    ################
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
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