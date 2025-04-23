from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    TrainingArguments,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from peft import AutoPeftModelForCausalLM, LoraConfig
import torch

checkpoint_dir = "qwen-r1-7b-xsum-dpo-beam6-1+beam6-2/summac"

if __name__=="__main__":
    model = AutoPeftModelForCausalLM.from_pretrained(f"{checkpoint_dir}/final_checkpoint", device_map="cuda", torch_dtype=torch.bfloat16) 
    model = model.merge_and_unload()   

    output_merged_dir = f"{checkpoint_dir}/final_merged_checkpoint"
    model.save_pretrained(output_merged_dir, safe_serialization=True)