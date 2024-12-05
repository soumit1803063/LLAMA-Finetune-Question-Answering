import os
import torch
import gc
import warnings
from dotenv import load_dotenv
from config import DATA_PATH,TEST_SIZE,RANDOM_STATE
from config import TRAINING_PROMPT_TEMPLATE, TRAINING_SYSTEM_PROMPT
from config import PREDICTION_PROMPT_TEMPLATE, PREDICTION_SYSTEM_PROMPT
from config import MODEL_PATH,FINETUNED_MODEL_PATH
from config import BATCH_SIZE,GRAD_ACCUM_STEPS,LOGGING_STEPS,LEARNING_RATE,NUM_EPOCHS,EVAL_STEPS
from config import MAX_NEW_TOKENS
from src.data_preparation.pipeline import data_preparation_pipeline
from src.prompt_preparation.pipeline import prompt_preparation_pipeline
from src.model_preparation.pipeline import model_preparation_pipeline
from src.fine_tune.pipeline import finetune_pipeline
from src.generate.pipeline import generation_pipeline
from src.evaluate.pipeline import evaluation_pipeline
from utils import login_to_hf


warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to: {DEVICE}")

load_dotenv()
HF_TOKEN = os.getenv("HF")
#login to hugging face
login_to_hf(HF_TOKEN)
torch.cuda.empty_cache()
gc.collect()

train_dataframe, test_dataframe = data_preparation_pipeline(data_file_path=DATA_PATH,
                                                          test_size=TEST_SIZE,
                                                          random_state=RANDOM_STATE)

print(f"Train Dataframe Shape = {train_dataframe.shape}.")
print(f"Test Dataframe Shape = {test_dataframe.shape}.")


train_prompts, test_prompts = prompt_preparation_pipeline(train_dataframe=train_dataframe,
                                                          train_prompt_template=TRAINING_PROMPT_TEMPLATE,
                                                          train_system_prompt=TRAINING_SYSTEM_PROMPT,
                                                          test_dataframe=test_dataframe,
                                                          test_prompt_template=PREDICTION_PROMPT_TEMPLATE,
                                                          test_system_prompt=PREDICTION_SYSTEM_PROMPT
                                                        )

print(f"Train Prompts = {len(train_prompts)}.")
print(f"Test Prompts = {len(test_prompts)}.")


training_model, training_tokenizer = model_preparation_pipeline(model_path=MODEL_PATH,
                                                                device=DEVICE)
print(f"Got Training Model.")
print(f"Got Training Tokenizer.")




finetune_pipeline(model=training_model,
                  tokenizer=training_tokenizer,
                  train_prompts=train_prompts,
                  val_prompts=test_prompts,
                  finetuned_model_dir= FINETUNED_MODEL_PATH,
                  batch_size= BATCH_SIZE,
                  grad_accum_steps=GRAD_ACCUM_STEPS,
                  logging_steps= LOGGING_STEPS,
                  learning_rate= LEARNING_RATE,
                  num_epochs= NUM_EPOCHS,
                  eval_steps= EVAL_STEPS)

torch.cuda.empty_cache()
gc.collect()



finetuned_model, finetuned_tokenizer = model_preparation_pipeline(model_path=FINETUNED_MODEL_PATH,
                                                                device=DEVICE)
print(f"Got Finetuned Model.")
print(f"Got Finetuned Tokenizer.")

generated_answers = generation_pipeline(prompts=test_prompts,
                                        model=finetuned_model,
                                        tokenizer=finetuned_tokenizer,
                                        max_new_tokens=MAX_NEW_TOKENS)

evaluation_result_dataframe = evaluation_pipeline(test_dataframe=test_dataframe,
                                                  generated_answers=generated_answers)