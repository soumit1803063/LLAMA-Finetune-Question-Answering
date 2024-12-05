import pandas as pd

def create_single_prompt(data_series: pd.Series, 
                         prompt_template: str, 
                         system_prompt: str, 
                         is_predict: bool = False) -> str:
    single_prompt = ""  # Initialize an empty string to store the modified single prompt
    
    if is_predict:
        # If is_predict is True, format the prompt without the answer
        single_prompt = prompt_template.format(
            system_prompt=system_prompt,  
            question=data_series["question"],  
            evidence_text=data_series["evidence_text"]  
        )
    else:
        # If is_predict is False, format the prompt with the answer
        single_prompt = prompt_template.format(
            system_prompt=system_prompt,  
            question=data_series["question"],  
            evidence_text=data_series["evidence_text"],  
            answer=data_series["answer"]  
        )
    
    return single_prompt

def create_prompts(dataframe: pd.DataFrame,  
                   prompt_template: str,  
                   system_prompt: str,  
                   is_predict: bool = False) -> list[str]:
    prompts = []  # Initialize an empty list to store the generated prompts
    
    for _, row in dataframe.iterrows():  # Iterate over each row in the DataFrame
        # Generate a single prompt
        single_prompt = create_single_prompt(data_series=row,  
                                             prompt_template=prompt_template,  
                                             system_prompt=system_prompt,  
                                             is_predict=is_predict)  
        prompts.append(single_prompt)  # Append the generated prompt to the list
    
    return prompts
