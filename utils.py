from huggingface_hub import login
def login_to_hf(token: str):
    # Logs into Hugging Face Hub using the provided token
    login(token=token)
    print("Logged in to Hugging Face Hub successfully.")
