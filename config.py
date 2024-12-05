# Paths
DATA_PATH = "data/syntheses_10.csv"  # Path to the dataset
MODEL_PATH = "NousResearch/Llama-2-7b-chat-hf"#"/media/somu/D/dataset/llama-2-pytorch-7b-chat-hf-v1"  # Path to the base model
FINETUNED_MODEL_PATH = "data/model/Fine_Tuned_LLaMA2"  # Path to the fine-tuned model output

# Split parameters for dataset
TEST_SIZE = 0.05  # Proportion of the dataset to include in the test split
RANDOM_STATE = 42  # Random state for reproducibility

# Prompts for training and prediction
TRAINING_SYSTEM_PROMPT = """You are a financial chatbot trained to answer questions based on the information provided.
Your responses should be directly sourced from the content of the evidence_text(context).
When asked a question, ensure that your answer is explicitly supported by the text and do not
include any external information, interpretations, or assumptions not clearly stated in the evidence_text(context).
If a question pertains to financial data or analysis that is not explicitly covered in the evidence_text(context) provided,
respond by stating that the information is not available in the evidence_text(context).
Your primary focus should be on accuracy, specificity, and adherence to the information in the evidence_text(context),
particularly regarding financial statements, company performance, and market positions."""

TRAINING_PROMPT_TEMPLATE = """
<s>[INST]
<<SYS>>
{system_prompt}
<</SYS>>
{question}
{evidence_text}
[/INST]
{answer}
</s>"
"""

PREDICTION_SYSTEM_PROMPT = """Give answer to questions provided below from the evidence text."""
PREDICTION_PROMPT_TEMPLATE = """
<s>[INST]
<<SYS>>
{system_prompt}
<</SYS>>

Here is the question:
{question}

Consider the provided text as evidence:
{evidence_text}
[/INST]
"""



# Training constants
BATCH_SIZE = 4  # Number of samples per batch
GRAD_ACCUM_STEPS = 4  # Gradient accumulation steps
LEARNING_RATE = 2e-5  # Learning rate for optimization
NUM_EPOCHS = 5  # Number of training epochs
EVAL_STEPS = 50  # Evaluation interval in steps
LOGGING_STEPS = 10  # Logging interval in steps
MAX_SEQ_LENGTH = 100  # Maximum sequence length for input data

#Prediction
MAX_NEW_TOKENS = 100

