import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Any

def get_bnb_config(load_in_4bit: bool = True,
                   bnb_4bit_use_double_quant: bool = True,
                   bnb_4bit_quant_type: str = "nf4",
                   bnb_4bit_compute_dtype: Any = torch.bfloat16
                   ) -> BitsAndBytesConfig:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,# Whether to load model in 4-bit precision
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant, # Whether to use double quantization
        bnb_4bit_quant_type=bnb_4bit_quant_type,# The quantization type (e.g., "nf4")
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype# The compute dtype (e.g., torch.bfloat16, torch.float16)
    )
    return bnb_config

def get_model(model_path: str,
              bnb_config: BitsAndBytesConfig,
              device:str):
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 quantization_config=bnb_config,
                                                 device_map = "auto")
    return model


def get_tokenizer(model_path: str, device: str) -> tuple:
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer