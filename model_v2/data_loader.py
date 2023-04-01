import numpy as np
import torch
import os

from torch.utils.data import DataLoader, Dataset, Subset

from utils import seed_worker

class TrafficData(Dataset):
    """
    Load data under folders
    """
    def __init__(self, args):
        self.args = args  
        
        X_path = f"{self.args.data_dir}/np_in_5min.npy"  # sequence features of TMC segments in frequency of 5 min
        Y_path = f"{self.args.data_dir}/np_out_5min.npy"  # ground truth of TMC speed & incident data in frequency of 5 min

        self.all_X = torch.from_numpy(np.load(X_path)).float()  # (21060, feat_dim)

        # (21060, num_seg, 4) the last dimension refers to 1. speed of all vehicles, 2. speed of truck, 3. speed of personal vehicles, 4. incident status
        self.Y = torch.from_numpy(np.load(Y_path)).float()  

        # change input based on whether to use new features or not
        new_feat_indices = []
        if args.use_dens:
            new_feat_indices.append(args.indices_dens)
        if args.use_spd_all:
            new_feat_indices.append(args.indices_spd_all)
        if args.use_spd_truck:
            new_feat_indices.append(args.indices_spd_truck)
        if args.use_spd_pv:
            new_feat_indices.append(args.indices_spd_pv)
        
        self.X = self.all_X[:, args.indices_spd_pv[1]:]  # (21060, 642)
        for i, j in new_feat_indices:
            self.X = torch.cat([self.all_X[:, i:j], self.X], axis=1)
        
    def __len__(self):
        return self.X.size(0) 

    def __getitem__(self, idx):
        x_idx_base = idx // 180
        x_idx_remain = min(max(idx % 180, 6, self.args.seq_len_in-1), np.floor(186 - self.args.seq_len_out)) # ensure we have valid idx based on input and output sequence length
        idx = int(x_idx_remain + x_idx_base * 180)
        Y_idx = [idx-6 + i for i in range(self.args.seq_len_out+1)]  # be careful, the starting point (first idx) of Y is the same as the last idx of X, and won't count into output sequence length
        
        X = self.X[(idx-self.args.seq_len_in+1):idx+1, :]
        Y = self.Y[Y_idx, :, :]

        return X, Y


def get_data_loader(args):
    """
    Creates training and testing data loaders for model training
    """
    whole_dataset = TrafficData(args=args)

    # train_size = int(np.ceil(args.data_train_ratio * len(whole_dataset)))
    # test_size = len(whole_dataset) - train_size

    # make sure the splitting preserves the integrity of 180 slots in a day
    train_size = int(np.ceil(args.data_train_ratio * (len(whole_dataset)/180))) * 180   # 14760
    val_size = int(np.ceil(args.data_val_ratio * (len(whole_dataset)/180))) * 180   # 4320
    test_size = len(whole_dataset) - train_size - val_size   # 1980

    # split train and test dataset
    # Option 1 - random split
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset = whole_dataset, lengths = [train_size, val_size, test_size], generator=g)

    # Option 2 - split by temporal order <= Actually, as demonstrated in the experiments, the distributions of trainding, validation and testing datasets seem different.
    # Note that we cannot directly slice whole_dataset by "whole_dataset[start:end]". We need to use torch.util.data.Subset instead.
    # train_dataset = Subset(whole_dataset, torch.arange(train_size))
    # val_dataset = Subset(whole_dataset, torch.arange(train_size,train_size+val_size))
    # test_dataset = Subset(whole_dataset, torch.arange(train_size+val_size, train_size+val_size+test_size))

    train_dloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=seed_worker)
    val_dloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=seed_worker)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=seed_worker)
    
    return train_dloader, val_dloader, test_dloader


def get_inference_data_loader(args):
    """
    Creates data loader for model inference
    """
    whole_dataset = TrafficData(args=args)
    whole_dloader = DataLoader(dataset=whole_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    return whole_dloader

def get_sorted_inference_data_loader(args):
    """
    FUNCTION:
        Creates data loader for model inference
        Output is sorted by time 

    OUTPUT:
        sorted_infer_dataloader
    """