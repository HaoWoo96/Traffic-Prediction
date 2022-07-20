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
        Y_path = f"{self.args.data_dir}/new_Y.npy"  # store true speed & incident data in frequency of 1 min  

        self.all_X = torch.from_numpy(np.load(X_path)).float()  # (29520, feat_dim)
        self.Y = torch.from_numpy(np.load(Y_path)).float()  # (5*29520, num_seg, 2) the last dimension refers to 1. speed and 2. incident status 

        # change input based on whether to use new features or not
        new_feat_indices = [args.density_indices, args.truck_spd_indices, args.pv_spd_indices]
        if not args.use_density:
            new_feat_indices.remove(args.density_indices)
        if not args.use_truck_spd:
            new_feat_indices.remove(args.truck_spd_indices)
        if not args.use_pv_spd:
            new_feat_indices.remove(args.pv_spd_indices)
        
        self.X = self.all_X[:, args.pv_spd_indices[1]:]  # 29520, 704
        for i, j in new_feat_indices:
            self.X = torch.cat([self.all_X[:, i:j], self.X], axis=1)
        
    def __len__(self):
        return self.X.size(0) 

    def __getitem__(self, idx):
        x_idx_base = idx // 180
        x_idx_remain = min(max(idx % 180, 6, self.args.in_seq_len-1), np.floor((930 - self.args.out_seq_len*self.args.out_freq)/5)) # ensure we have valid idx based on input and output sequence length
        idx = x_idx_remain + x_idx_base * 180 
        Y_idx = [(idx-6)*5 + i*self.args.out_freq for i in range(self.args.out_seq_len+1)]  # be careful, the starting point (first idx) of Y is the same as the last idx of X, and won't count into output sequence length

        '''
        recurrent:
            - denotes whether instance at index idx is a recurrent case or not
        '''
        recurrent = True  

        X = self.X[idx-self.args.in_seq_len+1:idx+1, :]
        Y = self.Y[Y_idx, :, :]

        if torch.sum(Y[:, :, 1]) > 0:
            # if there is/are incident(s) in output segments (sum of all entries is greater than 0)
            recurrent = False

        return X, Y, recurrent


class TrafficDataByCase(Dataset):
    '''
    dataset class for recurrent instances or nonrecurrent instances
    '''
    def __init__(self, X, Y):

        self.X = X  # (num_instance, in_seq_len, in_dim)
        self.Y = Y  # (num_instance, out_seq_len + 1, out_dim, 2)
    
    def __len__(self):
        return self.X.size(0)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], 0  # the third element is just a placeholder


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
    
    # create special dataloader for recurrent & non-recurrent prediction
    if args.task == "rec" or args.task == "nonrec":
        rec_X_path = f"{args.data_dir}/rec_X_{str(args.use_density)[0]}_{str(args.use_truck_spd)[0]}_{str(args.use_pv_spd)[0]}_{args.in_seq_len}_{args.out_seq_len}_{args.out_freq}.pt"
        rec_Y_path = f"{args.data_dir}/rec_Y_{str(args.use_density)[0]}_{str(args.use_truck_spd)[0]}_{str(args.use_pv_spd)[0]}_{args.in_seq_len}_{args.out_seq_len}_{args.out_freq}.pt"
        nonrec_X_path = f"{args.data_dir}/nonrec_X_{str(args.use_density)[0]}_{str(args.use_truck_spd)[0]}_{str(args.use_pv_spd)[0]}_{args.in_seq_len}_{args.out_seq_len}_{args.out_freq}.pt"
        nonrec_Y_path = f"{args.data_dir}/nonrec_Y_{str(args.use_density)[0]}_{str(args.use_truck_spd)[0]}_{str(args.use_pv_spd)[0]}_{args.in_seq_len}_{args.out_seq_len}_{args.out_freq}.pt"

        # create and save recurrent & non-recurrent data by splitting the whole dataset into 
        if not os.path.exists(rec_X_path):
            whole_dloader = DataLoader(dataset=whole_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            rec_X = torch.cat([x[recurrent==True] for x, y, recurrent in whole_dloader])
            rec_Y = torch.cat([y[recurrent==True] for x, y, recurrent in whole_dloader])
            torch.save(rec_X, rec_X_path)
            torch.save(rec_Y, rec_Y_path)

            nonrec_X = torch.cat([x[recurrent==False] for x, y, recurrent in whole_dloader])
            nonrec_Y = torch.cat([y[recurrent==False] for x, y, recurrent in whole_dloader])
            torch.save(nonrec_X, nonrec_X_path)
            torch.save(nonrec_Y, nonrec_Y_path)
        
        # load recurrent / nonrecurrent dataset and create dataloader
        if args.task == "rec":
            rec_X = torch.load(rec_X_path)
            rec_Y = torch.load(rec_Y_path)
            
            rec_dataset = TrafficDataByCase(rec_X, rec_Y)

            rec_train_size = int(np.ceil(args.train_ratio * len(rec_dataset)))
            rec_test_size = len(rec_dataset) - rec_train_size

            # split train and test dataset
            rec_train_dataset, rec_test_dataset = torch.utils.data.random_split(dataset = rec_dataset, lengths = [rec_train_size, rec_test_size], generator=torch.Generator().manual_seed(args.seed))

            rec_train_dloader = DataLoader(dataset=rec_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            rec_test_dloader = DataLoader(dataset=rec_test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

            return rec_train_dloader, rec_test_dloader
        
        else:
            nonrec_X = torch.load(nonrec_X_path)
            nonrec_Y = torch.load(nonrec_Y_path)
            
            nonrec_dataset = TrafficDataByCase(nonrec_X, nonrec_Y)

            nonrec_train_size = int(np.ceil(args.train_ratio * len(nonrec_dataset)))
            nonrec_test_size = len(nonrec_dataset) - nonrec_train_size

            # split train and test dataset
            nonrec_train_dataset, nonrec_test_dataset = torch.utils.data.random_split(dataset = nonrec_dataset, lengths = [nonrec_train_size, nonrec_test_size], generator=torch.Generator().manual_seed(args.seed))

            nonrec_train_dloader = DataLoader(dataset=nonrec_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            nonrec_test_dloader = DataLoader(dataset=nonrec_test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

            return nonrec_train_dloader, nonrec_test_dloader

    return train_dloader, test_dloader


def get_inference_data_loader(args):
    """
    Creates data loader for model inference
    """
    whole_dataset = TrafficData(args=args)
    whole_dloader = DataLoader(dataset=whole_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    return whole_dloader