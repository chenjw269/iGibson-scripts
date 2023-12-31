import pandas as pd

from utils import *

from torch import nn
from torch.utils.data import Dataset, DataLoader


class PosLocalMapWithoutObj(Dataset):
    
    def __init__(self, csv_pth):
        super(PosLocalMapWithoutObj, self).__init__()
        
        self.vocab = Vocab()
        self.embedding = nn.Embedding(512, 6)
        self.data = pd.read_csv(csv_pth)
        
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, index):
        
        # Read ground truth
        gt_pth = self.data["gt"][index]
        gt = read_gt(gt_pth)
        gt = gt[0, ...]
        
        # Read observation
        # obs_pth = self.data["rgb"][index]
        # obs = read_obs(obs_pth)
        
        # obs_pth = self.data["d"][index]
        # obs = read_depth(obs_pth)
        
        # Read map
        mp_pth = self.data["mp"][index]
        mp, mp_mask = read_obj_mp(mp_pth, self.vocab, self.embedding)
        
        
        return (gt, obs, mp, mp_mask)