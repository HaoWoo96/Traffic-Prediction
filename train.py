import numpy as np
import argparse
import torch
import torch.optim as optim
import logging
import sys

from torch.utils.tensorboard import SummaryWriter

from models import TrafficSeq2Seq, TrafficModel
from data_loader import get_data_loader
from utils import save_checkpoint, create_dir


def train(train_dataloader, model, opt, epoch, args, writer):

    model.train()
    step = epoch*len(train_dataloader)
    epoch_loss = 0

    for i, batch in enumerate(train_dataloader):
        x, target, _ = batch
        x = x.to(args.device)  # (batch_size, in_seq_len, in_dim)
        target = target.to(args.device)  # (batch_size, out_seq_len + 1, out_dim, 2) for TrafficModel, (batch_size, out_seq_len + 1, out_dim) for TrafficSeq2Seq

        # Forward Pass and Compute Loss
        if args.task == "LR":
            # the second element returned are attention weights, which won't be used here but will be visualized during inference.
            pred, _ = model(x, target)
            criterion = torch.nn.BCEWithLogitsLoss()  # combine nn.Sigmoid() with nn.BCELoss() but more numerically stable
            loss = criterion(pred, target[:, 1:, :, 1])
        else:
            if args.task == "finetune":
                # the second element returned are incident predictions (logits), the third element returned are attention weights, which won't be used here but will be visualized during inference.
                pred, inc_pred, _ = model(x, target) 
            else:
                # the second element returned are attention weights, which won't be used here but will be visualized during inference.
                pred, _ = model(x, target)
        
        epoch_loss += loss

        # Backward and Optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Logging
        writer.add_scalar("train_loss", loss.item(), step+i)
    
    return epoch_loss/len(train_dataloader)


def test(test_dataloader, model, epoch, args, writer):

    model.eval()
    step = epoch*len(test_dataloader)
    epoch_loss = 0

    for i, batch in enumerate(test_dataloader):
        x, target, _ = batch
        x = x.to(args.device)  # (batch_size, in_seq_len, in_dim)
        target = target.to(args.device)  # (batch_size, out_seq_len + 1, out_dim, 2) for TrafficModel, (batch_size, out_seq_len + 1, out_dim) for TrafficSeq2Seq

        with torch.no_grad():
            # Make Prediction and Compute Loss
            if args.task == "LR":
                pred, _ = model(x, target)
                criterion = torch.nn.BCEWithLogitsLoss()  # combine nn.Sigmoid() with nn.BCELoss() but more numerically stable
                loss = criterion(pred, target[:, 1:, :, 1])
            else:
                if args.task == "finetune":
                    pred, inc_pred, _ = model(x, target) 
                else:
                    pred, _ = model(x, target)
                criterion = torch.nn.MSELoss()
                loss = criterion(pred, target[:, 1:, :, 0])
            epoch_loss += loss

        # Logging
        writer.add_scalar("test_loss", loss.item(), step+i)
    
    return epoch_loss/len(test_dataloader)


def main(args):
    """
    train model, evaluate on test data, and save checkpoints
    """
    # 1. Create Directories
    create_dir(args.checkpoint_dir)
    create_dir(args.log_dir)

    # 2. Set up Logger for Tensorboard and Logging
    writer = SummaryWriter('{}/{}'.format(args.log_dir,args.exp_name))
    if args.load_checkpoint_epoch > 0:
        logging.basicConfig(filename=f"{args.log_dir}/{args.exp_name}/training_resume.log", filemode="w", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG) 
    else:
        logging.basicConfig(filename=f"{args.log_dir}/{args.exp_name}/training.log", filemode="w", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG) 

    # log experiment and data information
    logging.info('{:*^100}'.format(" COMMAND LINE "))
    logging.info(" ".join(sys.argv) + "\n")

    logging.info('{:*^100}'.format(" EXPERIMENT INFORMATION "))
    logging.info(f"Task: {args.task}")
    logging.info(f"Experiment Name: {args.exp_name}")
    logging.info(f"Number of Epochs: {args.num_epochs}, Learning Rate: {args.lr}, Batch Size: {args.batch_size} \n")

    logging.info('{:*^100}'.format(" DATA INFORMATION "))
    logging.info(f"Use Density: {args.use_density}; Use Truck Speed: {args.use_truck_spd}; Use Personal Vehicle Speed: {args.use_pv_spd}")
    logging.info(f"Input Sequence Length: {args.in_seq_len}; Output Sequence Lenth: {args.out_seq_len}; Output Frequency: {args.out_freq} \n")

    logging.info('{:*^100}'.format(" LOADING PROGRESS "))

    # 3. Initialize Model
    if args.task != "base":
        model = TrafficModel(args).to(args.device)
    else:
        model = TrafficSeq2Seq(args).to(args.device)

    # 4. Load Checkpoint 
    if args.load_checkpoint:
        model_path = "{}/{}.pt".format(args.checkpoint_dir, args.load_checkpoint)
        with open(model_path, 'rb') as f:
            state_dict = torch.load(f, map_location=args.device)
            model.load_state_dict(state_dict)
        logging.info(f"successfully loaded checkpoint from {model_path}")
    else:
        # in "rec" and "nonrec" tasks, if checkpoint is not specified, 
        # then we need to initialize the model with best checkpoint from "LR"
        if (args.task == "rec" or args.task == "nonrec"):
            model_path = "/".join(args.checkpoint_dir.split("/")[:-1]) + "/LR/best_{}.pt".format(args.exp_name)
            with open(model_path, 'rb') as f:
                state_dict = torch.load(f, map_location=args.device)
                model.load_state_dict(state_dict)
            logging.info(f"successfully loaded checkpoint from {model_path}")

        # in "finetune" task, if checkpoint is not specified,
        # then we need to initialize the model with best checkpoint from "LR" (encoder & LR_decoder), "rec" (rec_decoder) and "nonrec" (nonrec_decoder)
        elif args.task == "finetune":
            LR_model_path = "/".join(args.checkpoint_dir.split("/")[:-1]) + "/LR/best_{}.pt".format(args.exp_name)
            rec_model_path = "/".join(args.checkpoint_dir.split("/")[:-1]) + "/rec/best_{}.pt".format(args.exp_name)
            nonrec_model_path = "/".join(args.checkpoint_dir.split("/")[:-1]) + "/nonrec/best_{}.pt".format(args.exp_name)

            # load state dict partially to initialize corresponding modules of TrafficModel
            with open(LR_model_path, 'rb') as f_LR, open(rec_model_path, 'rb') as f_rec, open(nonrec_model_path, 'rb') as f_nonrec:
                
                # load state dict 
                state_dict_LR = torch.load(f_LR, map_location=args.device)
                state_dict_rec = torch.load(f_rec, map_location=args.device)
                state_dict_nonrec = torch.load(f_nonrec, map_location=args.device)

                # retain corresponding modules only 
                enc_dec_lr_state_d = {k:v for k, v in state_dict_LR.items() if "LR" in k or "encoder" in k}
                dec_rec_state_d = {k:v for k, v in state_dict_rec.items() if "rec" in k and "nonrec" not in k}
                dec_nonrec_state_d = {k:v for k, v in state_dict_nonrec.items() if "nonrec" in k}

                # load state dict of corresponding modules 
                model.load_state_dict(enc_dec_lr_state_d, strict=False)
                model.load_state_dict(dec_rec_state_d, strict=False)
                model.load_state_dict(dec_nonrec_state_d, strict = False)

            logging.info(f"successfully loaded checkpoint from \
                    {LR_model_path} \n\
                    {rec_model_path} \n\
                    {nonrec_model_path} ")

    # 5. Freeze Module 
    if args.task == "LR":
        # in "LR" task, freeze decoders for recurrent and nonrecurrent prediction
        model.rec_decoder.requires_grad_(False)
        model.nonrec_decoder.requires_grad_(False)
    elif args.task == "rec":
        # in "rec" task, freeze everything except recurrent decoder
        model.encoder.requires_grad_(False)
        model.LR_decoder.requires_grad_(False)
        model.nonrec_decoder.requires_grad_(False)
    elif args.task == "nonrec":
        # in "nonrec" task, freeze everythign except nonrecurrent decoder
        model.encoder.requires_grad_(False)
        model.LR_decoder.requires_grad_(False)
        model.rec_decoder.requires_grad_(False)

    # 6. Set up Optimizer
    opt = optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999))

    # 7. Load Data for Training & Testing
    train_dataloader, test_dataloader = get_data_loader(args=args)
    logging.info(f"successfully loaded data \n")

    # 8. Train, Test & Save Checkpoints
    # Logging
    if args.load_checkpoint_epoch > 0:
        logging.info('{:=^100}'.format(" Training Resumes from Epoch {} ".format(args.load_checkpoint_epoch)))
    else:
        logging.info('{:=^100}'.format(" Training Starts "))
    logging.info("please check tensorboard for plots of experiment {}/{}".format(args.log_dir, args.exp_name))
    logging.info("please check logging messages at {}/{}/training.log".format(args.log_dir, args.exp_name))
    
    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(max(0, args.load_checkpoint_epoch), args.num_epochs):
        # Train
        train_epoch_loss = train(train_dataloader, model, opt, epoch, args, writer)
        test_epoch_loss = test(test_dataloader, model, epoch, args, writer)

        logging.info("epoch: {}   train loss (per batch): {:.4f}   test loss (per batch): {:.4f}".format(epoch, train_epoch_loss, test_epoch_loss))
        
        # Save Model Checkpoint Regularly
        if epoch % args.checkpoint_every == 0:
            logging.info("checkpoint saved at epoch {}".format(epoch))
            save_checkpoint(epoch=epoch, model=model, args=args, best=False)

        # Save Best Model Checkpoint
        if (test_epoch_loss <= best_loss):
            best_loss = test_epoch_loss
            best_epoch = epoch
            logging.info("best model saved at epoch {}".format(epoch))
            save_checkpoint(epoch=epoch, model=model, args=args, best=True)

    logging.info('{:=^100}'.format(" Training Completes ({} epochs trained, best epoch at {} with test loss per batch = {:.4f}) ".format(args.num_epochs, best_epoch, best_loss)))


def create_parser():
    """
    Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # 1. Model hyper-parameters
    parser.add_argument('--num_layer', type=int, default=2, help='Number of stacked GRUs in encoder and decoder')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension in encoder and decoder')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help='threshold of teacher forcing')

    parser.add_argument('--in_seq_len', type=int, default=7, help='sequence length of input')
    parser.add_argument('--out_seq_len', type=int, default=6, help='sequence length of output')
    parser.add_argument('--out_freq', type=int, default=5, help='frequency of output data')

    # 2. Data Hyper-parameters
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of training data versus whole data')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random splitting')

    parser.add_argument('--in_dim', type=int, default = 1403, help='dimension of input')
    parser.add_argument('--out_dim', type=int, default=78, help=' dimension of output i.e. number of segments (78 by default)')

    parser.add_argument('--density_indices', type=list or tuple, default = [0, 233], help='[start_idx, end_idx], indices of density features')
    parser.add_argument('--truck_spd_indices', type=list or tuple, default = [233, 466], help='[start_idx, end_idx], indices of truck speed features')
    parser.add_argument('--pv_spd_indices', type=list or tuple, default = [466, 699], help='[start_idx, end_idx], indices of personal vehicle speed features')
    parser.add_argument('--incident_indices', type=list or tuple, default = [1166, 1382], help='[start_idx, end_idx], indices of incident features')
    
    parser.add_argument('--use_density', action="store_true", help='use density features or not')
    parser.add_argument('--use_truck_spd', action="store_true", help='use truck speed features or not')
    parser.add_argument('--use_pv_spd', action="store_true", help='use personal vehicle speed features or not')
    
    # fow now we don't use embedding for incident and density, as they are ordinal variables and are already embedded in new_X.npy
    # therefore, we don't use the following arguments 
    # parser.add_argument('--in_feat_dim', type=int, help='numeric_feat_dim + incident_feat_dim * incident_embed_dim')
    # parser.add_argument('--incident_feat_dim', type=int, help='length of incident features')
    # parser.add_argument('--incident_range', type=int, default=3, help='3 by default ("0", "1", "2")')
    # parser.add_argument('--incident_embed_dim', type=int, help='dimension of embeded incident features')
    # parser.add_argument('--numeric_feat_dim', type=int, help='length of numercial features')

    # 3. Training Hyper-parameters
    '''
    TASKs:
        1. "LR": call train_LR() for logistic regression, train encoder and LR_decoder
        2. "rec": call train_rec() for speed prediction, freeze encoder, train rec_decoder
        3. "nonrec": call_train_nonrec() for speed prediction, freeze encoder, train nonrec_decoder
        4. "finetune": call_finetune() for speed prediction, load checkpoint of encoder, LR_decoder, rec_decoder and nonrec_decoder, and finetune them together
        5. “base": 
            - train TrafficSeq2Seq model
            - input: no new features
            - output: in 5-min frequency
    '''
    parser.add_argument('--task', type=str, help="Choose one from 'LR', 'rec', 'nonrec', 'finetune', 'base'")

    parser.add_argument('--num_epochs', type=int, default=401, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of sequences in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default 0.001)')

    parser.add_argument('--exp_name', type=str, default="exp", help='Name of the experiment')

    # 4. Directories and Checkpoint/Sample Iterations
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--log_dir', type=str, default='./logs')

    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_every', type=int , default=10)

    parser.add_argument('--load_checkpoint', type=str, default='', help='Name of the checkpoint')
    parser.add_argument('--load_checkpoint_epoch', type=int, default=-1, help='Epoch of the checkpoint')

    return parser


if __name__ == '__main__':
    
    # 1. Modify Arguments
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # For reproducibility
    torch.manual_seed(args.seed)

    # Task specific directories
    args.log_dir += f"/{args.task}"
    args.checkpoint_dir += f"/{args.task}" 
    args.exp_name += f"_{str(args.use_density)[0]}_{str(args.use_truck_spd)[0]}_{str(args.use_pv_spd)[0]}_{args.in_seq_len}_{args.out_seq_len}_{args.out_freq}" 
    if args.load_checkpoint_epoch > 0:
        args.load_checkpoint = f"epoch_{args.load_checkpoint_epoch}_{args.exp_name}"

    # Change input dimension based on task type and whether to use new features or not
    if not args.use_density:
        args.in_dim -= 233
    
    if not args.use_truck_spd:
        args.in_dim -= 233
    
    if not args.use_pv_spd:
        args.in_dim -= 233


    # 2. Execute Training Pipeline
    main(args)