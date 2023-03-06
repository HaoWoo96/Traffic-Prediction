import numpy as np
import torch
import os

from torch.utils.data import DataLoader, Dataset

class TrafficData(Dataset):
    """
    Load data under folders
    """
    def __init__(self, args):
        self.args = args  
        
        X_path = f"{self.args.data_dir}/new_X.npy"  # store sequence features frequency of 5 min
        Y_path = f"{self.args.data_dir}/new_Y_{args.gt_type}.npy"  # store true speed & incident data in frequency of 1 min (XD) or 5 min (TMC)

        self.all_X = torch.from_numpy(np.load(X_path)).float()  # (29520, feat_dim)

        # For new_Y_xd: (5*29520, num_seg, 2) the last dimension refers to 1. speed and 2. incident status
        # For new_Y_tmc: (29520, num_seg, 4), the last dimension refers to 1. all speed, 2. truck speed, 3. personal vehicle speed, and 4. incident status
        self.Y = torch.from_numpy(np.load(Y_path)).float()   

        # change input based on whether to use new features or not
        new_feat_indices = []
        if args.use_density:
            new_feat_indices.append(args.density_indices)
        if args.use_speed:
            new_feat_indices.append(args.speed_indices)
        if args.use_truck_spd:
            new_feat_indices.append(args.truck_spd_indices)
        if args.use_pv_spd:
            new_feat_indices.append(args.pv_spd_indices)
        
        self.X = self.all_X[:, args.pv_spd_indices[1]:]  # 29520, 704
        for i, j in new_feat_indices:
            self.X = torch.cat([self.all_X[:, i:j], self.X], axis=1)
        
    def __len__(self):
        return self.X.size(0) 

    def __getitem__(self, idx):
        x_idx_base = idx // 180
        if self.args.gt_type == "tmc":
            x_idx_remain = min(max(idx % 180, 6, self.args.in_seq_len-1), np.floor(186 - self.args.out_seq_len)) # ensure we have valid idx based on input and output sequence length
            idx = int(x_idx_remain + x_idx_base * 180)
            Y_idx = [idx-6 + i for i in range(self.args.out_seq_len+1)]  # be careful, the starting point (first idx) of Y is the same as the last idx of X, and won't count into output sequence length
        else:
            x_idx_remain = min(max(idx % 180, 6, self.args.in_seq_len-1), np.floor((186*5 - self.args.out_seq_len*self.args.out_freq)/5)) # ensure we have valid idx based on input and output sequence length
            idx = int(x_idx_remain + x_idx_base * 180)
            Y_idx = [(idx-6)*5 + i*self.args.out_freq for i in range(self.args.out_seq_len+1)]  # be careful, the starting point (first idx) of Y is the same as the last idx of X, and won't count into output sequence length
 

        X = self.X[(idx-self.args.in_seq_len+1):idx+1, :]
        Y = self.Y[Y_idx, :, :]

        return X, Y


def get_data_loader(args):
    """
    Creates training and testing data loaders for model training
    """
    whole_dataset = TrafficData(args=args)

    train_size = int(np.ceil(args.train_ratio * len(whole_dataset)))
    test_size = len(whole_dataset) - train_size

    # split train and test dataset
    train_dataset, test_dataset = torch.utils.data.random_split(dataset = whole_dataset, lengths = [train_size, test_size], generator=torch.Generator().manual_seed(args.seed))

    train_dloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    
    return train_dloader, test_dloader


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