import numpy as np
import argparse
import torch
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from models import TrafficSeq2Seq, TrafficModel
from data_loader import get_data_loader
from utils import save_checkpoint, create_dir

def train(train_dataloader, model, opt, epoch, args, writer):

    model.train()
    step = epoch*len(train_dataloader)
    epoch_loss = 0

    for i, batch in enumerate(train_dataloader):
        input, target = batch

        input = input.to(args.device)  # (batch_size, in_seq_len, numeric_feat_dim + incident_feat_dim)
        target = target.to(args.device)  # (batch_size, out_seq_len + 1, out_dim, 2) for TrafficModel, (batch_size, out_seq_len + 1, out_dim) for TrafficSeq2Seq

        # Forward Pass
        pred = model(input, target)

        # Compute Loss
        if args.task == "LR":
            criterion = torch.nn.BCELoss()
            loss = criterion(pred, target[:, 1:, :, 1])
        else:
            criterion = torch.nn.MSELoss()
            if args.task == "base":
                loss = criterion(pred, target[:, 1:, :])
            else:
                loss = criterion(pred, target[:, 1:, :, 0])
        epoch_loss += loss

        # Backward and Optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Logging
        writer.add_scalar("train_loss", loss.item(), step+i)
    
    return epoch_loss


def test(test_dataloader, model, epoch, args, writer):

    model.eval()
    step = epoch*len(test_dataloader)
    epoch_loss = 0

    for i, batch in enumerate(test_dataloader):
        input, target = batch

        input = input.to(args.device)  # (batch_size, in_seq_len, numeric_feat_dim + incident_feat_dim)
        target = target.to(args.device)  # (batch_size, out_seq_len + 1, out_dim, 2) for TrafficModel, (batch_size, out_seq_len + 1, out_dim) for TrafficSeq2Seq

        # Make Prediction
        with torch.no_grad():
            pred = model(input, target)

            # Compute Loss
            if args.task == "LR":
                criterion = torch.nn.BCELoss()
                loss = criterion(pred, target[:, 1:, :, 1])
            else:
                criterion = torch.nn.MSELoss()
                if args.task == "base":
                    loss = criterion(pred, target[:, 1:, :])
                else:
                    loss = criterion(pred, target[:, 1:, :, 0])
            epoch_loss += loss

        # Logging
        writer.add_scalar("test_loss", loss.item(), step+i)
    
    return epoch_loss


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


def create_parser():
    """
    Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--num_layer', type=int, default=2, help='Number of stacked GRUs in encoder and decoder')
    parser.add_argument('--hidden_num', type=int, default=256, help='Hidden dimension in encoder and decoder')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help='threshold of teacher forcing')

    parser.add_argument('--in_seq_len', type=int, default=7, help='sequence length of input')
    parser.add_argument('--out_seq_len', type=int, default=5, help='sequence length of output')
    parser.add_argument('--out_freq', type=int, default=5, help='frequency of output data')


    # Data Hyper-parameters
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of training data versus whole data')

    parser.add_argument('--incident_indices', type=list or tuple, help='[start_idx, end_idx], indices of categorical features (incident)')
    parser.add_argument('--incident_feat_dim', type=int, help='length of incident features')
    parser.add_argument('--incident_range', type=int, default=3, help='3 by default ("0", "1", "2")')
    parser.add_argument('--incident_embed_dim', type=int, help='dimension of embeded incident features')
    parser.add_argument('--conf_indices', type=list or tuple, help='[start_idx, end_idx], indices of categorical (ordinal) features ()')

    parser.add_argument('--numeric_feat_dim', type=int, help='length of numercial features')
    parser.add_argument('--in_feat_dim', type=int, help='numeric_feat_dim + incident_feat_dim * incident_embed_dim')
    parser.add_argument('--out_dim', type=int, default=162, help=' dimension of output i.e. number of segments (162 by default)')


    # Training Hyper-parameters
    parser.add_argument('--task', type=str, help="Choose one from 'LR', 'rec', 'nonrec', 'finetune', 'base'")
    # 1. "LR": call train_LR() for logistic regression, train encoder and LR_decoder
    # 2. "rec": call train_rec() for speed prediction, freeze encoder, train rec_decoder
    # 3. "nonrec": call_train_nonrec() for speed prediction, freeze encoder, train nonrec_decoder
    # 4. "finetune": call_finetune() for speed prediction, load checkpoint of encoder, LR_decoder, rec_decoder and nonrec_decoder, and finetune them together
    # 5. “base": train TrafficSeq2Seq model 

    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of sequences in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default 0.001)')

    parser.add_argument('--exp_name', type=str, default="exp", help='Name of the experiment')


    # Directories and Checkpoint/Sample Iterations
    parser.add_argument('--main_dir', type=str, default='./data/')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_every', type=int , default=10)

    parser.add_argument('--load_checkpoint', type=str, default='', help='Name of the checkpoint')

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    args.checkpoint_dir = args.checkpoint_dir+"/"+args.task # checkpoint directory is task specific

    main(args)