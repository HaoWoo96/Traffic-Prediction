import numpy as np
import argparse
import torch
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from models import TrafficSeq2Seq, TrafficModel
from data_loader import get_data_loader
from utils import save_checkpoint, create_dir

'''
args:
    # dataset
    train_ratio: split ratio from 0 to 1
    batch_size

    # sequence length & frequency
    in_seq_len: sequence length of input (default 7)
    out_seq_len: sequence length of output (default 5)
    out_freq: frequency of output data (default 5)

    # features
    incident_indices: [start_idx, end_idx], indices of categorical features (incident)
    incident_feat_dim: length of incident features
    incident_range: 3 by default ("0", "1", "2")
    incident_embed_dim: dimension of embeded incident features
    conf_indices: [start_idx, end_idx], indices of categorical (ordinal) features ()

    numeric_feat_dim: length of mnumercial features
    in_feat_dim: numeric_dim + incident_feat_dim * incident_embed_dim

    in_seq_len: sequence length of input (default 7)
    out_seq_len: sequence length of prediction output (default 5), not including the starting point
    out_freq: frequency of output data (default 5)

    out_dim: dimension of output i.e. number of segments (162 by default)

    # model 
    num_layer: number of layers of GRUs in encoder & decoder (2 by default)
    hidden_dim: hidden dim of GRUs in encoder & decoder

    teacher_forcing_ratio: threshold of teacher forcing

    # training
    device
    exp_name
    checkpoint_every: the frequency of saving checkpoint
    
    num_epochs: number of epochs for training 

    main_dir
    checkpoint_every
    checkpoint_dir
    load_checkpoint: str, name of checkpoint
    task: 
        1. "LR": call train_LR() for logistic regression, train encoder and LR_decoder
        2. "rec": call train_rec() for speed prediction, freeze encoder, train rec_decoder
        3. "nonrec": call_train_nonrec() for speed prediction, freeze encoder, train nonrec_decoder
        4. "finetune": call_finetune() for speed prediction, load checkpoint of encoder, LR_decoder, rec_decoder and nonrec_decoder, and finetune them together
        5. “base": train TrafficSeq2Seq model 
'''

def train()

def test()

def main(args):
    """
    train encoder and LR_decoder of TrafficModel
    """
    # Create Directories
    create_dir(args.checkpoint_dir)
    create_dir('./logs')

    # Tensorboard Logger
    writer = SummaryWriter('./logs/{}_{}'.format(args.task, args.exp_name))

    # Initialize Model
    if args.task != "baseline":
        model = TrafficModel(args)
    else:
        model = TrafficSeq2Seq(args)
    
    # Load Checkpoint 
    if (args.task == "rec" or args.task == "nonrec") and not args.load_checkpoint:
        # in "rec" and "nonrec", if checkpoint is not specified, 
        # then we need to initialize the model with best checkpoint from "LR"
        args.checkpoint_dir = "LR"
        args.load_checkpoint = "best"

    if args.load_checkpoint:
        model_path = "{}/{}.pt".format(args.checkpoint_dir, args.load_checkpoint)
        with open(model_path, 'rb') as f:
            state_dict = torch.load(f, map_location=args.device)
            model.load_state_dict(state_dict)
        print ("successfully loaded checkpoint from {}".format(model_path))

    if args.task == "finetune" and not args.load_checkpoint:
        # in "finetune", if checkpoint is not specified,
        # then we need to initialize teh model with best checkpoint from "LR" (encoder & LR_decoder), "rec" (rec_decoder) and "nonrec" (nonrec_decoder)
        LR_model_path = "LR/best.pt"
        rec_model_path = "rec/best.pt"
        nonrec_model_path = "nonrec/best.pt"

        # load state dict partially to initialize corresponding modules of TrafficModel
        # =============== TO DO ===================
        # with open(model_path, 'rb') as f:
        #     state_dict = torch.load(f, map_location=args.device)
        #     model.load_state_dict(state_dict)
        # print ("successfully loaded checkpoint from {}".format(model_path))
    
    # Freeze Module 
    if args.task == "LR":
        model.rec_decoder.requires_grad_(False)
        model.nonrec_decoder.requires_grad_(False)
    elif args.task == "rec":
        model.encoder.requires_grad_(False)
        model.LR_decoder.requires_grad_(False)
        model.nonrec_decoder.requires_grad_(False)
    elif args.task == "nonrec":
        model.encoder.requires_grad_(False)
        model.LR_decoder.requires_grad_(False)
        model.rec_decoder.requires_grad_(False)

    # Optimizer
    opt = optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999))

    # Dataloader for Training & Testing
    train_dataloader, test_dataloader = get_data_loader(args=args)

    print ("successfully loaded data")

    best_loss = float("inf")

    print ("======== start training for {} task ========".format(args.task))
    print ("(check tensorboard for plots of experiment logs/{}_{})".format(args.task, args.exp_name))
    
    for epoch in range(args.num_epochs):

        # Train
        train_epoch_loss = train(train_dataloader, model, opt, epoch, args, writer)
        test_epoch_loss = test(test_dataloader, model, epoch, args, writer)

        print ("epoch: {}   train loss: {:.4f}   test loss: {:.4f}".format(epoch, train_epoch_loss, test_epoch_loss))
        
        # Save Model Checkpoint Regularly
        if epoch % args.checkpoint_every == 0:
            print ("checkpoint saved at epoch {}".format(epoch))
            save_checkpoint(epoch=epoch, model=model, args=args, best=False)

        # Save Best Model Checkpoint
        if (test_epoch_loss >= best_loss):
            best_loss = test_epoch_loss
            print ("best model saved at epoch {}".format(epoch))
            save_checkpoint(epoch=epoch, model=model, args=args, best=True)

    print ("======== training completes ========")