from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments
from trl import SFTTrainer
import pandas as pd
from datasets import Dataset

def get_lora_config(r: int = 16,
                    lora_alpha: int = 64,
                    target_modules: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"],
                    lora_dropout: float = 0.1,
                    bias: str = "none",
                    task_type: str = "CAUSAL_LM"
                    ) -> LoraConfig:
    lora_config = LoraConfig(
        r=r,# The rank of the low-rank decomposition
        lora_alpha=lora_alpha,# Scaling factor for the low-rank matrix
        target_modules=target_modules,# Target modules (e.g., LLaMA-specific layers)
        lora_dropout=lora_dropout,# Dropout rate for the low-rank layers
        bias=bias,# Bias term ("none", "all", or "lora_only")
        task_type=task_type # Task type (e.g., "CAUSAL_LM")
    )
    return lora_config

def apply_lora(model,lora_config):
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    return model


def get_training_args(
    output_dir: str,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    logging_steps: int,
    learning_rate: float,
    num_train_epochs: int,
    eval_steps: int,
    seed: int = 42,
    optim: str = "paged_adamw_32bit",
    fp16: bool = True,
    weight_decay: float = 0.01,
    max_grad_norm: float = 0.3,
    evaluation_strategy: str = "steps",
    warmup_ratio: float = 0.05,
    save_strategy: str = "epoch",
    group_by_length: bool = True,
    lr_scheduler_type: str = "cosine",
    push_to_hub: bool = True,
) -> TrainingArguments:

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        num_train_epochs=num_train_epochs,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        warmup_ratio=warmup_ratio,
        save_strategy=save_strategy,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        seed=seed,
        push_to_hub=push_to_hub,
    )
    return training_args


def train_model(model,
                tokenizer,
                lora_config,
                training_args,
                train_prompts,
                val_prompts,
                max_seq_length: int = 100):
    
    train_dataset = Dataset.from_pandas(pd.DataFrame({"text": train_prompts}))
    val_dataset = Dataset.from_pandas(pd.DataFrame({"text": val_prompts}))
    # Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )
    
    # Train the model
    trainer.train()