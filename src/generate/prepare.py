
def extract_answer(generated_text):
    # Extract the Answer Portion from the whole generated text
    answer_start = generated_text.find("[/INST]") + len("[/INST]")  # Find the end of </INST> tag
    answer = generated_text[answer_start:].strip()  # Extract everything after that position
    return answer    


def generate(prompt,model,tokenizer,max_new_tokens: int = 100):
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate response
    output = model.generate(input_ids=inputs["input_ids"], max_new_tokens=max_new_tokens)

    # Decode the response
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text