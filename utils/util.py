import gc
import torch
from keybert import KeyBERT

global_kw_model = None
is_cuda_available = torch.cuda.is_available()

def get_keyword_model():
    """lazy load the keyword model
    """
    global global_kw_model
    if global_kw_model is None:
        global_kw_model = KeyBERT()
        global_kw_model.to('cuda' if is_cuda_available else 'cpu')
        
    return global_kw_model


def free_keyword_model():
    global global_kw_model
    if global_kw_model is not None:
        del global_kw_model
        global_kw_model = None
        gc.collect()
        torch.cuda.empty_cache()
