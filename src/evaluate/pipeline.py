import pandas as pd
from .metrics import calculate_cosine_similarity

def evaluation_pipeline(test_dataframe, generated_answers):
    # Initialize a list to store the results
    results = []

    for idx, generated_answer in enumerate(generated_answers):
        original_answer = test_dataframe.iloc[idx, 1]  # Assuming the original answer is in the second column
        cos_similarity = calculate_cosine_similarity(original_answer, generated_answer)

        # Append the result as a dictionary
        results.append({
            "Original Answer": original_answer,
            "Generated Answer": generated_answer,
            "Cosine Similarity": cos_similarity
        })

    # Convert the results list to a DataFrame
    result_df = pd.DataFrame(results)
    return result_df
