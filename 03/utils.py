import os
from PIL import Image

import torch


def read_gt(gt_pth):
    
    traj = os.listdir(gt_pth)
    
    traj_gt = torch.Tensor()
    
    for idx in range(len(traj)):

        gt_f = f"{gt_pth}/{traj[idx]}"
        gt_fp = open(gt_f, "r", encoding="utf-8")
        gt_data = gt_fp.readline()
        gt_data = gt_data.replace("\n", "").split(",")
        gt_data = [
            round(float(gt_data[0]), 4),
            round(float(gt_data[1]), 4),
            round(float(gt_data[2]), 4)
        ]
        pos_gt = torch.Tensor(gt_data).unsqueeze(0)    
        traj_gt = torch.concat((traj_gt, pos_gt), dim=0)
    

    return traj_gt


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


import os
import numpy as np
from PIL import Image

import torch


def read_rgb(obs_pth):
    
    traj = os.listdir(obs_pth)
    
    obs = torch.Tensor()
    
    for idx in range(int(len(traj) / 8)):

        pos_obs = torch.Tensor()
        
        for d in range(8):
            
            obs_item = f"{obs_pth}/{str(idx).zfill(4)}_{d}.jpg"
            obs_image = (Image.open(obs_item).convert("RGB")).resize((224, 224))
            obs_ts = torch.Tensor(np.array(obs_image)).permute(2, 0, 1)
            obs_ts = obs_ts.unsqueeze(0)

            pos_obs = torch.concat((pos_obs, obs_ts))

        pos_obs = pos_obs.unsqueeze(0)
        obs = torch.concat((obs, pos_obs))

    
    return obs


import os
import numpy as np
from PIL import Image

import torch


def read_depth(obs_pth):
    
    traj = os.listdir(obs_pth)
    
    obs = torch.Tensor()
    
    for idx in range(int(len(traj) / 8)):

        pos_obs = torch.Tensor()
        
        for d in range(8):
            
            obs_item = f"{obs_pth}/{str(idx).zfill(4)}_{d}.jpg"
            obs_image = (Image.open(obs_item).convert("L")).resize((224, 224))
            obs_ts = torch.Tensor(np.array(obs_image)).unsqueeze(0).repeat(3, 1, 1)
            obs_ts = obs_ts.unsqueeze(0)

            pos_obs = torch.concat((pos_obs, obs_ts))

        pos_obs = pos_obs.unsqueeze(0)
        obs = torch.concat((obs, pos_obs))

    
    return obs


import torch


def read_mp(mp_pth):
    
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