from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch

class TrafficData(Dataset):
    """
    Load data under folders
    """
    def __init__(self, args):
        self.args = args  

        if args.task == "base":
            X_path = self.args.main_dir  + "/X_base.npy"  # store sequence features frequency of 5 min
            Y_path = self.args.main_dir  + "/Y_base.npy"  # store true speed data in frequency of 5 min
        else:
            X_path = self.args.main_dir  + "/X.npy"  # store sequence features frequency of 5 min
            Y_path = self.args.main_dir  + "/Y.npy"  # store true speed & incident data in frequency of 1 min  
            

        self.X = torch.from_numpy(np.load(X_path))  # (29520, feat_dim)
        self.Y = torch.from_numpy(np.load(Y_path))  # (5*29520, num_seg, 2) the last dimension refers to 1. speed and 2. incident status 
        
    def __len__(self):
        return self.X.size(0) 

    def __getitem__(self, idx):
        l = len(self)
        idx = min(max(idx, self.args.in_seq_len-1), np.floor((l-self.args.out_seq_len*self.args.out_freq)/5))  # ensure we have valid idx based on input and output sequence length
        Y_idx = [idx*5 + i*self.args.out_freq for i in range(self.args.out_seq_len+1)]  # be careful, the starting point (first idx) of Y is the same as the last idx of X, and won't count into output sequence length
        return self.X[idx-self.args.in_seq_len+1:idx+1], self.Y[Y_idx, :, :]


def get_data_loader(args):
    """
    Creates training and test data loaders
    """
    whole_dataset = TrafficData(args=args)

    train_size = np.ceil(args.train_ratio * len(whole_dataset))
    test_size = len(whole_dataset) - train_size

    # split train and test dataset
    train_dataset, test_dataset = torch.utils.data.random_split(dataset = whole_dataset, lengths = [train_size, test_size], generator=torch.Generator().manual_seed(args.seed))

    train_dloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    

    return train_dloader, test_dloader