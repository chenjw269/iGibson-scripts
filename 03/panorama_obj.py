from dataset.igibson_utils import *
# from igibson_utils import *

import pandas as pd

from torch.utils.data import Dataset, DataLoader


import torch
from torch import nn


def read_obj_mp(mp_pth, vocab, embedding):
    """_summary_

    Args:
        mp_pth (_type_): _description_
        vocab (_type_): _description_
        embedding (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Read map data
    mp_fp = open(mp_pth, "r", encoding="utf-8")
    mp_data = mp_fp.readlines()
    mp_data = [
        (i.replace("\n", "")).split(",") for i in mp_data
    ]
    mp_fp.close()
    
    mp_ts_label = []
    mp_ts_geometric = []
    
    for idx in range(len(mp_data)):
        
        mp_ts_label += [vocab.add(mp_data[idx][0])]
        mp_ts_geometric += [
            round(float(mp_data[idx][1]), 4),
            round(float(mp_data[idx][2]), 4),
            round(float(mp_data[idx][3]), 4),
            round(float(mp_data[idx][4]), 4),
            round(float(mp_data[idx][5]), 4),
            round(float(mp_data[idx][6]), 4),
        ]
    
    # Embed semantic label
    mp_ts_label = torch.Tensor(mp_ts_label)
    mp_ts_label = embedding(mp_ts_label)
        
    # Add Gaussian noise
    mp_ts_geometric = torch.Tensor(mp_ts_geometric)

    mean = 0
    std = 0.03
    mp_noise = torch.randn(mp_ts_geometric.shape) * std + mean
    mp_ts_geometric = mp_ts_geometric + mp_noise
    
    # Adjust the map tensor length
    mp_data = torch.concat((mp_ts_label, mp_ts_geometric))
    
    mp_len = len(mp_data)
    mp_dim = len(mp_data[0])
    
    ad_mp_len = 20
    
    mp_mask_1 = torch.ones(mp_len)
    
    if mp_len < ad_mp_len:
        ad_mp_data = torch.zeros((ad_mp_len - mp_len, mp_dim))    
        mp_data = torch.concat((mp_data, ad_mp_data))

        mp_mask_2 = torch.zeros(ad_mp_len - mp_len)    
        mp_mask = torch.concat((mp_mask_1, mp_mask_2))
    
    
    return mp_data, mp_mask