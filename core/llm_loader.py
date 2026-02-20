# core/llm_loader.py

import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from .config import MODEL_NAME, MAX_NEW_TOKENS


class LLMFactory:

    def __init__(self):
        self.model_name = MODEL_NAME

    def load(self):

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=nf4_config,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_safetensors=True
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
        )

        model_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            device_map="auto"
        )

        return HuggingFacePipeline(pipeline=model_pipeline)