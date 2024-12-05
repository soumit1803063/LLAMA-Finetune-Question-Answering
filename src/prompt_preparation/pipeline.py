from .prepare import create_prompts
import pandas as pd
def prompt_preparation_pipeline(train_dataframe: pd.DataFrame,
                                train_prompt_template: str,
                                train_system_prompt: str,
                                test_dataframe: pd.DataFrame,
                                test_prompt_template: str,
                                test_system_prompt: str,
                                ):
    train_prompts = create_prompts(dataframe=train_dataframe,
                                   prompt_template=train_prompt_template,
                                   system_prompt=train_system_prompt,
                                   is_predict=False)
    
    test_prompts = create_prompts(dataframe=test_dataframe,
                                   prompt_template=test_prompt_template,
                                   system_prompt=test_system_prompt,
                                   is_predict=True)
    
    return train_prompts, test_prompts