from .prepare import generate,extract_answer
def generation_pipeline(prompts,
                               model,
                               tokenizer,
                               max_new_tokens):
        generated_answers = []
        for idx, prompt in enumerate(prompts):
                generated_text = generate(prompt=prompt,
                                          model=model,
                                          tokenizer=tokenizer,
                                          max_new_tokens=max_new_tokens)
                generated_answer = extract_answer(generated_text=generated_text)
                generated_answers.append(generated_answer)
        return generated_answers
