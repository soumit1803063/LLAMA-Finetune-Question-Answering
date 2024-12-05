from .prepare import get_bnb_config,get_model,get_tokenizer

def model_preparation_pipeline(model_path:str,
                               device:str
                                ):
    
    bnb_config = get_bnb_config()
    model = get_model(model_path=model_path,
                      bnb_config=bnb_config,
                      device=device)
    tokenizer = get_tokenizer(model_path=model_path,
                              device=device)
    
    return model, tokenizer
    