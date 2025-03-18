import gc
import torch
import numpy as np

from keybert import KeyBERT
from keybert.backend import BaseEmbedder
from sentence_transformers import SentenceTransformer

from typing import Dict

global_kw_model = None
is_cuda_available = torch.cuda.is_available()

class CustomEmbedder(BaseEmbedder):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model

        # all available CUDA devices will be used
        self.pool = self.embedding_model.start_multi_process_pool()

    def embed(self, documents, verbose=False):

        # Run encode() on multiple GPUs
        embeddings = self.embedding_model.encode_multi_process(documents, 
                                                               self.pool)
        return embeddings


def get_keyword_model():
    """lazy load the keyword model
    """
    global global_kw_model

    if global_kw_model is None:
        model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
        custom_embedder = CustomEmbedder(embedding_model=model)
        global_kw_model = KeyBERT(model = custom_embedder)
        # global_kw_model.to('cuda' if is_cuda_available else 'cpu')
        
    return global_kw_model


def free_keyword_model():
    global global_kw_model
    if global_kw_model is not None:
        del global_kw_model
        global_kw_model = None
        gc.collect()
        torch.cuda.empty_cache()




def save_np_dict(edge_dict:Dict[str, np.array], path:str):
    """_summary_

    Parameters
    ----------
    edge_dict : Dict[str, np.array]
        {"src_table-des_table": np.array} 2-D array
    path : str
        file path to save the edges
    """
    np.savez(path, **edge_dict)


def load_np_dict(path:str) -> Dict[str, np.array]:
    """_summary_

    Parameters
    ----------
    path : str
        file path to load the edges

    Returns
    -------
    Dict[str, np.array]
        {"src_table-des_table": np.array} 2-D array
    """
    loaded = np.load(path, allow_pickle=True)
    return {key:loaded[key] for key in loaded.files}