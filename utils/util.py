import gc
import torch
import numpy as np
import pandas as pd

from relbench.base import Table
from keybert import KeyBERT
from keybert.backend import BaseEmbedder
from sentence_transformers import SentenceTransformer

from typing import Dict, Any







def get_default_text_embedder_cfg():
    pass

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

