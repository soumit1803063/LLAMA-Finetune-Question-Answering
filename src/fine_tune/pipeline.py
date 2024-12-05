from .prepare import get_lora_config, apply_lora, get_training_args, train_model

def finetune_pipeline(model,
                      tokenizer,
                      train_prompts,
                      val_prompts,
                      finetuned_model_dir: str,
                      batch_size: int,
                      grad_accum_steps: int,
                      logging_steps: int,
                      learning_rate: float,
                      num_epochs: int,
                      eval_steps: int,

                      max_seq_length: int = 100,
                      optim: str = "paged_adamw_32bit",
                      fp16: bool = True,
                      weight_decay: float = 0.01,
                      max_grad_norm: float = 0.3,
                      evaluation_strategy: str = "steps",
                      warmup_ratio: float = 0.05,
                      save_strategy: str = "epoch",
                      group_by_length: bool = True,
                      lr_scheduler_type: str = "cosine",
                      push_to_hub: bool = True,):
    

    lora_config = get_lora_config()
    lora_applied_model = apply_lora(model=model, lora_config=lora_config)

    trainable_params = sum(p.numel() for p in lora_applied_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lora_applied_model.parameters())
    print(f"Trainable params: {trainable_params} || Total params: {total_params} || Trainable%: {100 * trainable_params / total_params:.2f}%")

    training_args = get_training_args(output_dir=finetuned_model_dir,
                                      per_device_train_batch_size=batch_size,
                                      gradient_accumulation_steps=grad_accum_steps,
                                      logging_steps=logging_steps,
                                      learning_rate=learning_rate,
                                      num_train_epochs=num_epochs,
                                      eval_steps=eval_steps,
                                      optim=optim,
                                      fp16=fp16,
                                      weight_decay=weight_decay,
                                      max_grad_norm=max_grad_norm,
                                      evaluation_strategy=evaluation_strategy,
                                      warmup_ratio=warmup_ratio,
                                      save_strategy=save_strategy,
                                      group_by_length=group_by_length,
                                      lr_scheduler_type=lr_scheduler_type,
                                      push_to_hub=push_to_hub)
    train_model(model=lora_applied_model,
                tokenizer=tokenizer,
                lora_config=lora_config,
                training_args=training_args,
                train_prompts=train_prompts,
                val_prompts=val_prompts,
                max_seq_length=max_seq_length)
