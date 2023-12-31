import os
from PIL import Image

import torch


def read_gt_position(gt_pth):
    """Read ground truth position

    Args:
        gt_pth (str): ground truth file path

    Returns:
        torch.Tensor: ground truth value
    """
    gt_f = os.listdir(gt_pth)[0]
    
    gt_f = f"{gt_pth}/{gt_f}"
    gt_fp = open(gt_f, "r", encoding="utf-8")
    gt_data = gt_fp.readline()
    gt_data = gt_data.replace("\n", "").split(",")
    gt_data = [
        round(float(gt_data[0]), 4),
        round(float(gt_data[1]), 4),
        round(float(gt_data[2]), 4)
    ]
    gt_ts = torch.Tensor(gt_data)


    return gt_ts


import os
from PIL import Image
import numpy as np

import torch


def read_d_position(d_pth):
    """Read depth image

    Args:
        d_pth (str): depth image path
    """
    d_list = os.listdir(d_pth)
    
    d_ts = torch.Tensor()
    
    for idx in range(len(d_list)):
        
        d_item = f"{d_pth}/0000_{idx}.jpg"
        d = (Image.open(d_item).convert("L")).resize((224, 224))
        d = torch.Tensor(np.array(d)).unsqueeze(0).repeat(3, 1, 1)
        d = d.unsqueeze(0)
        
        d_ts = torch.concat((d_ts, d))
        
    return d_ts


import os
from PIL import Image
import numpy as np

import torch


def read_rgb_position(rgb_pth):
    """Read rgb image

    Args:
        rgb_pth (str): rgb image path
    """
    rgb_list = os.listdir(rgb_pth)
    
    rgb_ts = torch.Tensor()
    
    for idx in range(len(rgb_list)):
        
        rgb_item = f"{rgb_pth}/0000_{idx}.jpg"
        rgb = (Image.open(rgb_item).convert("L")).resize((224, 224))
        rgb = torch.Tensor(np.array(rgb)).unsqueeze(0).repeat(3, 1, 1)
        rgb = rgb.unsqueeze(0)
        
        rgb_ts = torch.concat((rgb_ts, rgb))
        
    return rgb_ts


def read_mp_image(mp_pth):
    """Read map image

    Args:
        mp_pth (str): map image path
    """
    mp = (Image.open(mp_pth).convert("L")).resize((224, 224))
    mp = torch.Tensor(np.array(mp)).unsqueeze(0).repeat(3, 1, 1)

    return mp


def read_mp_layout(mp_pth):
    """Read layout map in semantic object format
    
    Args:
        mp_pth (str): layout map file path
    
    Returns:
        torch.Tensor: layout map value
    """    
    # Read map data
    mp_fp = open(mp_pth, "r", encoding="utf-8")
    mp_data = mp_fp.readlines()
    mp_data = [
        (i.replace("\n", "")).split(",") for i in mp_data
    ]
    mp_fp.close()
    
    for idx in range(len(mp_data)):
        mp_data[idx] = [
            round(float(mp_data[idx][0]), 4),
            round(float(mp_data[idx][1]), 4),
            round(float(mp_data[idx][2]), 4),
            round(float(mp_data[idx][3]), 4),
        ]
    
    mp_data = torch.Tensor(mp_data)
    
    # Add Gaussian noise
    mean = 0
    std = 0.03
    mp_noise = torch.randn(mp_data.shape) * std + mean
    mp_data = mp_data + mp_noise
    
    # Adjust the map tensor length    
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


""" Vocabulary for semantic label """

class Vocab:
    """
    Class of vocabulary
    """
    def __init__(self):
        
        self.word_to_index = {}
        self.index_to_word = []
        
    def add(self, token):
        """Add a word to vocabulary

        Args:
            token (str): the word to add

        Returns:
            int: the word index in vocabulary
        """
        token = token.lower()
        if token in self.index_to_word:
            pass
        else:
            self.index_to_word.append(token)
            self.word_to_index[token] = len(self.index_to_word)
            
        return self.word_to_index[token]
    
    
def read_mp_obj(mp_pth, vocab, embedding):
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
        
        mp_ts_label += [int(vocab.add(mp_data[idx][0]))]
        mp_ts_geometric.append([
            round(float(mp_data[idx][1]), 4),
            round(float(mp_data[idx][2]), 4),
            round(float(mp_data[idx][3]), 4),
            round(float(mp_data[idx][4]), 4),
            round(float(mp_data[idx][5]), 4),
            round(float(mp_data[idx][6]), 4),
        ])
    
    # Embed semantic label
    mp_ts_label = torch.Tensor(mp_ts_label).to(dtype=torch.int)
    mp_ts_label = embedding(mp_ts_label)
        
    # Add Gaussian noise
    mp_ts_geometric = torch.Tensor(mp_ts_geometric)

    mean = 0
    std = 0.03
    mp_noise = torch.randn(mp_ts_geometric.shape) * std + mean
    mp_ts_geometric = mp_ts_geometric + mp_noise
    
    # Adjust the map tensor length
    mp_data = torch.concat((mp_ts_label, mp_ts_geometric), dim=1)
    
    mp_len = len(mp_data)
    mp_dim = len(mp_data[0])
    
    ad_mp_len = 150
    
    mp_mask_1 = torch.ones(mp_len)
    
    if mp_len < ad_mp_len:
        ad_mp_data = torch.zeros((ad_mp_len - mp_len, mp_dim))    
        mp_data = torch.concat((mp_data, ad_mp_data))

        mp_mask_2 = torch.zeros(ad_mp_len - mp_len)    
        mp_mask = torch.concat((mp_mask_1, mp_mask_2))
    
    
    return mp_data, mp_mask