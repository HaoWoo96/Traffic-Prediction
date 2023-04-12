import numpy as np
import argparse
import torch
import torch.optim as optim
import logging
import os
import pickle
import sys

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime as dt

from models import *
from data_loader import get_data_loader
from utils import seed_torch, compute_train_time, save_checkpoint, create_dir, log_train_meta

########################################
#       TRAINING/TESTING FUNCTIONS     #
########################################

def train(train_dataloader, model, opt, epoch, args, writer):

    model.train()
    step = epoch*len(train_dataloader)
    epoch_loss = 0
    num_sample = 0

    for i, batch in enumerate(train_dataloader):
        x, target = batch
        x = x.to(args.device)  # (batch_size, in_seq_len, dim_in)
        target = target.to(args.device)  # (batch_size, out_seq_len + 1, dim_out, 2 or 4) for TrafficModel, (batch_size, out_seq_len + 1, dim_out) for TrafficSeq2Seq

        # Forward Pass
        if args.model_type == "Seq2Seq":
            # the second element returned are incident predictions (logits) in finetune task, or hidden tensor in other tasks
            # the third element returned are attention weights, which won't be used here but will be visualized during inference.
            pred, _, _ = model(x, target, mode="train")
        elif args.model_type == "Trans":
            pred = model(x, target, mode="train")
        else:
            pred = model(x, target, mode="train")

        # Compute Loss
        if args.task == "LR":
            # combine nn.Sigmoid() with nn.BCELoss() but more numerically stable
            # don't forget to set pos_weight
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.LR_pos_weight))
            loss_per_sample = criterion(pred, target[:, 1:, :, 3])  # avg loss per time step per sample
        else: 
            criterion = torch.nn.MSELoss()
            # register hook to only consider certain segments for the computation of loss
            if args.task in {"rec", "nonrec"}:
                if args.task == "rec":
                    h = pred.register_hook(lambda grad: grad * (target[:, 1:, :, 3] < 0.5).float())
                else:
                    h = pred.register_hook(lambda grad: grad * (target[:, 1:, :, 3] >= 0.5).float())
            loss_per_sample = criterion(pred, target[:, 1:, :, 0])  # avg loss per time step per sample

        epoch_loss += loss_per_sample*x.size(0)
        num_sample += x.size(0)

        # Backward and Optimize
        opt.zero_grad()
        loss_per_sample.backward()
        opt.step()

        # remove hook if needed
        if args.task in {"rec", "nonrec"}:
            h.remove()

        # Logging
        writer.add_scalar("train_loss", loss_per_sample.item(), step+i)
    
    # Decay Learning Rate
    # scheduler.step()
    
    # return epoch_loss/len(train_dataloader)
    return epoch_loss/num_sample  # avg loss per time step per sample


def eval(eval_dataloader, model, epoch, args, writer, eval_for_validation):

    model.eval() # deactivate dropout, and adjust for batch normalization
    step = epoch*len(eval_dataloader)
    epoch_loss = 0
    num_sample = 0

    for i, batch in enumerate(eval_dataloader):
        x, target = batch
        x = x.to(args.device)  # (batch_size, in_seq_len, dim_in)
        target = target.to(args.device)  # (batch_size, out_seq_len + 1, dim_out, 2) for TrafficModel, (batch_size, out_seq_len + 1, dim_out) for TrafficSeq2Seq

        with torch.no_grad():
            # Forward Pass
            if args.model_type == "Seq2Seq":
                # the second element returned are incident predictions (logits) in finetune task, or hidden tensor in other tasks
                # the third element returned are attention weights, which won't be used here but will be visualized during inference.
                pred, _, _ = model(x, target, mode="eval")
            elif args.model_type == "Trans":
                pred = model(x, target, mode="eval")
            else:
                pred = model(x, target)

            # Compute Loss
            if args.task == "LR":
                # combine nn.Sigmoid() with nn.BCELoss() but more numerically stable
                # don't forget to set pos_weight
                criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.LR_pos_weight))
                loss_per_sample = criterion(pred, target[:, 1:, :, 3])  # avg loss per time step per sample
            else: 
                criterion = torch.nn.MSELoss()
                loss_per_sample = criterion(pred, target[:, 1:, :, 0])

            epoch_loss += loss_per_sample*x.size(0)
            num_sample += x.size(0)

        # Logging
        if eval_for_validation:
            writer.add_scalar("val_loss", loss_per_sample.item(), step+i)
        else:
            writer.add_scalar("test_loss", loss_per_sample.item(), step+i)
    
    # return epoch_loss/len(eval_dataloader)
    return epoch_loss/num_sample  # avg loss per time step per sample


########################################
#           TRAINING PIPELINE          #
########################################
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
        logging.basicConfig(filename=f"{args.log_dir}/{args.exp_name}/training.log", filemode="a", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG) 
    else:
        logging.basicConfig(filename=f"{args.log_dir}/{args.exp_name}/training.log", filemode="w", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG) 
        # log meta info of training experiment
        log_train_meta(args)

    # 3. Initialize Model
    if args.task == "no_fact":
        if args.model_type == "Seq2Seq":
            model = Seq2SeqNoFact(args).to(args.device)
        elif args.model_type == "Trans":
            model = TransNoFact(args).to(args.device)
        else:
            model = STGCNNoFact(args).to(args.device)
    elif args.task == "naive":
        if args.model_type == "Seq2Seq":
            model = Seq2SeqFactNaive(args).to(args.device)
        elif args.model_type == "Trans":
            model = TransFactNaive(args).to(args.device)
        else:
            model = STGCNFactNaive(args).to(args.device)
    elif args.task == "naive_2enc":
        if args.model_type == "Seq2Seq":
            model = Seq2SeqFactNaive_2enc(args).to(args.device)
        else:
            pass
    else:
        if args.model_type == "Seq2Seq":
            model = Seq2SeqFact(args).to(args.device)
        elif args.model_type == "Trans":
            model = TransFact(args).to(args.device)
        else:
            model = STGCNFact(args).to(args.device)

    # 4. Set up Optimizer and LR Scheduler
    opt = optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999))
    # scheduler = optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_freq, gamma=args.lr_decay_rate)


    logging.info('{:*^100}'.format(" LOADING PROGRESS "))

    # 5. Load Checkpoint 
    if args.load_checkpoint:
        checkpoint_path = "{}/{}.pt".format(args.checkpoint_dir, args.load_checkpoint)
        with open(checkpoint_path, 'rb') as f:
            state_dict = torch.load(f, map_location=args.device)
            model.load_state_dict(state_dict["model"])
            opt.load_state_dict(state_dict["optimizer"])
        logging.info(f"successfully loaded checkpoint from {checkpoint_path}")
    else:
        # in "rec" and "nonrec" tasks, if checkpoint is not specified, 
        # then we need to initialize the model with best checkpoint from "LR"
        if (args.task == "rec" or args.task == "nonrec"):
            checkpoint_path = "/".join(args.checkpoint_dir.split("/")[:-1]) + "/LR/best_{}.pt".format(args.exp_name)
            with open(checkpoint_path, 'rb') as f:
                state_dict = torch.load(f, map_location=args.device)
                model.load_state_dict(state_dict["model"])
            logging.info(f"successfully loaded checkpoint from {checkpoint_path}")

        # in "finetune" task, if checkpoint is not specified,
        # then we need to initialize the model with best checkpoint from "LR" (encoder & LR_decoder), "rec" (rec_decoder) and "nonrec" (nonrec_decoder)
        elif args.task == "finetune":
            # LR_model_path = "/".join(args.checkpoint_dir.split("/")[:-1]) + "/LR/best_{}.pt".format(args.exp_name)
            # rec_model_path = "/".join(args.checkpoint_dir.split("/")[:-1]) + "/rec/best_{}.pt".format(args.exp_name)
            # nonrec_model_path = "/".join(args.checkpoint_dir.split("/")[:-1]) + "/nonrec/best_{}.pt".format(args.exp_name)

            # LR/Rec/Nonrec modules are trained with args.use_expectation, although it doesn't make any difference whether args.use_expectation is true or not
            # Therefore, pretrained LR/Rec/Nonrec modules are all marked with "best_exp_x_x_x_x_x_x_T.pt".
            # args.use_expectation does make a difference in finetune task
            LR_model_path = "/".join(args.checkpoint_dir.split("/")[:-1]) + "/LR/best_{}_T.pt".format("_".join(args.exp_name.split("_")[:-1]))  
            rec_model_path = "/".join(args.checkpoint_dir.split("/")[:-1]) + "/rec/best_{}_T.pt".format("_".join(args.exp_name.split("_")[:-1]))
            nonrec_model_path = "/".join(args.checkpoint_dir.split("/")[:-1]) + "/nonrec/best_{}_T.pt".format("_".join(args.exp_name.split("_")[:-1]))

            # load state dict partially to initialize corresponding modules of TrafficModel
            with open(LR_model_path, 'rb') as f_LR, open(rec_model_path, 'rb') as f_rec, open(nonrec_model_path, 'rb') as f_nonrec:
                
                # load state dict 
                state_dict_LR = torch.load(f_LR, map_location=args.device)
                state_dict_rec = torch.load(f_rec, map_location=args.device)
                state_dict_nonrec = torch.load(f_nonrec, map_location=args.device)

                # retain corresponding modules only 
                enc_dec_lr_state_d = {k:v for k, v in state_dict_LR["model"].items() if "LR" in k or "encoder" in k}
                dec_rec_state_d = {k:v for k, v in state_dict_rec["model"].items() if "rec" in k and "nonrec" not in k}
                dec_nonrec_state_d = {k:v for k, v in state_dict_nonrec["model"].items() if "nonrec" in k}

                # load state dict of corresponding modules 
                model.load_state_dict(enc_dec_lr_state_d, strict=False)
                model.load_state_dict(dec_rec_state_d, strict=False)
                model.load_state_dict(dec_nonrec_state_d, strict = False)

            logging.info(f"successfully loaded checkpoint from \
                    {LR_model_path} \n\
                    {rec_model_path} \n\
                    {nonrec_model_path} ")

    # 6. Freeze Module 
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

    # 7. Load Data for Training & Testing
    train_dataloader, val_dataloader, test_dataloader = get_data_loader(args=args)
    logging.info(f"successfully loaded data \n")

    # 8. Train, Test & Save Checkpoints
    # Logging
    if args.load_checkpoint_epoch > 0:
        logging.info('{:=^100}'.format(" Training Resumes from Epoch {} ".format(args.load_checkpoint_epoch)))
    else:
        logging.info('{:=^100}'.format(" Training Starts "))
        logging.info("please check tensorboard for plots of experiment {}/{}".format(args.log_dir, args.exp_name))
        logging.info("please check logging messages at {}/{}/training.log".format(args.log_dir, args.exp_name))
    logging.info(f"Average losses are calculated per time step per sample.")
    
    start_time = dt.now()
    best_val_loss = float("inf")
    best_test_loss = float("inf")
    if args.load_checkpoint_epoch > 0:
        checkpoint_path = "{}/{}.pt".format(args.checkpoint_dir, args.load_checkpoint)
        with open(checkpoint_path, 'rb') as f:
            state_dict = torch.load(f, map_location=args.device)
            best_val_loss = state_dict["losses"]["best_val_epoch_loss_hitherto"]
    best_epoch = -1  # evaluated based on validation loss

    for epoch in range(max(0, args.load_checkpoint_epoch+1), args.num_epochs):
        # Train
        train_epoch_loss = train(train_dataloader, model, opt, epoch, args, writer)
        val_epoch_loss = eval(val_dataloader, model, epoch, args, writer, True)
        test_epoch_loss = eval(test_dataloader, model, epoch, args, writer, False)

        logging.info("epoch: {}   train loss: {:.4f}   val loss: {:.4f}   test loss: {:.4f}".format(epoch, train_epoch_loss, val_epoch_loss, test_epoch_loss))
        
        # Save Model Checkpoint Regularly
        if epoch % args.checkpoint_every == 0:
            logging.info("checkpoint saved at epoch {}".format(epoch))
            save_checkpoint(epoch=epoch, model=model, opt=opt, args=args, train_epoch_loss=train_epoch_loss, val_epoch_loss=val_epoch_loss, test_epoch_loss=test_epoch_loss, best_val_epoch_loss_hitherto=best_val_loss, best=False)

        # Save Best Model Checkpoint
        if (val_epoch_loss <= best_val_loss):
            best_val_loss = val_epoch_loss
            best_test_loss = test_epoch_loss
            best_epoch = epoch
            logging.info("best model saved at epoch {}".format(epoch))
            save_checkpoint(epoch=epoch, model=model, opt=opt, args=args, train_epoch_loss=train_epoch_loss, val_epoch_loss=val_epoch_loss, test_epoch_loss=test_epoch_loss, best_val_epoch_loss_hitherto=best_val_loss, best=True)

    end_time = dt.now()
    training_time = compute_train_time(start_time, end_time)
    logging.info('{:=^100}'.format(" Training completes after {} hr {} min {} sec ({} epochs trained, best epoch at {} with val loss = {:.4f} and test loss = {:.4f}) ".format(training_time["hours"], training_time["minutes"], training_time["seconds"], args.num_epochs, best_epoch, best_val_loss, best_test_loss)))


def create_parser():
    """
    Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # 1. Model hyper-parameters
    parser.add_argument('--model_type', type=str, help="Choose one from 'Seq2Seq', 'Trans', 'STGCN'")

    parser.add_argument('--dim_hidden', type=int, default=256, help='Hidden dimension in encoder and decoder')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help='threshold of teacher forcing')
    parser.add_argument('--dropout_prob', type=float, default=0.1, help='dropout probability')

    parser.add_argument('--num_layer_GRU', type=int, default=2, help='Number of stacked GRUs in encoder and decoder')

    parser.add_argument('--num_head', type=int, default=8, help='Number of heads in a transformer encoder/decoder layer')
    parser.add_argument('--num_layer_Trans', type=int, default=2, help='Number of transformer encoder/decoder layers')

    parser.add_argument('--seq_len_in', type=int, default=7, help='sequence length of input')
    parser.add_argument('--seq_len_out', type=int, default=6, help='sequence length of output')
    parser.add_argument('--freq_out', type=int, default=5, help='frequency of output data')

    parser.add_argument('--inc_threshold', type=float, default=0.5, help='threshold of a prediction be considered as an incident')
    parser.add_argument('--LR_pos_weight', type=float, default=0.1840, help='ratio of positive samples in incident ground truth')
    parser.add_argument('--use_expectation', action="store_true", help='use expectation of speed prediction as model output')

    # Assuming perfect incident status prediction at stage 1 of the 2-stage model (Traffic)
    parser.add_argument('--use_gt_inc', type=int, default=0, help='use ground truth of indicent status as input to second stage')

    # 2. Data Hyper-parameters
    parser.add_argument('--data_train_ratio', type=float, default=0.7, help='Ratio of training data versus whole data')
    parser.add_argument('--data_val_ratio', type=float, default=0.2, help='Ratio of validation data versus whole data')
    parser.add_argument('--seed', type=int, default=912, help='Seed for random splitting')
    # parser.add_argument('--gt_type', type=str, default="tmc", help='ground truth speed type, "tmc" or "xd"')

    parser.add_argument('--dim_in', type=int, default=1470, help='dimension of input')
    parser.add_argument('--dim_out', type=int, default=207, help=' dimension of output i.e. number of segments (207 by default)')

    parser.add_argument('--indices_dens', type=list or tuple, default = [0, 207], help='[start_idx, end_idx], indices of density features')
    parser.add_argument('--indices_spd_all', type=list or tuple, default = [207, 414], help='[start_idx, end_idx], indices of speed features')
    parser.add_argument('--indices_spd_truck', type=list or tuple, default = [414, 621], help='[start_idx, end_idx], indices of truck speed')
    parser.add_argument('--indices_spd_pv', type=list or tuple, default = [621, 828], help='[start_idx, end_idx], indices of personal vehicle speed')
    # parser.add_argument('--indices_inc', type=list or tuple, default = [1035, 1242], help='[start_idx, end_idx], indices of incident status features')
    
    parser.add_argument('--use_dens', action="store_true", help='use density features or not')
    parser.add_argument('--use_spd_all', action="store_true", help='use speed data of <all vehicles> or not')
    parser.add_argument('--use_spd_truck', action="store_true", help='use speed data of <truck> or not')
    parser.add_argument('--use_spd_pv', action="store_true", help='use speed data of <personal vehicles> or not')
    
    # fow now we don't use embedding for incident and density, as they are ordinal variables and are already embedded in data processing
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
        5. "no_fact": 
            - train a model without Factorization
            - input: no new features
            - output: in 5-min frequency
        6. "naive": naive combination of three encoder-decoder modules (LR, rec, nonrec)
        7. "naive_2enc": naive combination of two-encoder-three-decoder modules (LR, rec, nonrec)
    '''
    parser.add_argument('--task', type=str, help="Choose one from 'LR', 'rec', 'nonrec', 'finetune', 'no_fact', 'naive', 'naive_2enc")

    parser.add_argument('--num_epochs', type=int, default=601, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of sequences in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate (default 0.0003)')
    # parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='Decay Rate of Learning rate (default 0.1)')
    # parser.add_argument('--lr_decay_freq', type=int, default=50, help='Decay Frequency (in terms of epochs) of Learning rate  (default 50)')

    parser.add_argument('--exp_name', type=str,help='Name of the experiment')

    # 4. Directories and Checkpoint/Sample Iterations
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--log_dir', type=str, default='./logs')

    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_every', type=int , default=20)

    parser.add_argument('--load_checkpoint', type=str, default='', help='Name of the checkpoint')
    parser.add_argument('--load_checkpoint_epoch', type=int, default=-1, help='Epoch of the checkpoint')

    return parser


if __name__ == '__main__':
    
    # 1. Modify Arguments
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # For reproducibility
    seed_torch(args.seed)

    # Task specific directories
    args.log_dir += f"/{args.task}"
    args.checkpoint_dir += f"/{args.task}" 

    args.exp_name = args.model_type
    args.exp_name += f"_{str(args.use_dens)[0]}_{str(args.use_spd_all)[0]}_{str(args.use_spd_truck)[0]}_{str(args.use_spd_pv)[0]}_{args.seq_len_in}_{args.seq_len_out}_{args.freq_out}_hidden_{args.dim_hidden}_batch_{args.batch_size}_{str(args.use_expectation)[0]}" 
    if not args.use_expectation:
        args.exp_name += f"_{str(args.inc_threshold)}"

    if args.load_checkpoint_epoch > 0:
        args.load_checkpoint = f"epoch_{args.load_checkpoint_epoch}_{args.exp_name}"

    # Change input dimension based on task type and whether to use new features or not
    if not args.use_dens:
        args.dim_in -= 207 
    if not args.use_spd_all:
        args.dim_in -= 207
    if not args.use_spd_truck:
        args.dim_in -= 207
    if not args.use_spd_pv:
        args.dim_in -= 207

    # 2. Execute Training Pipeline
    main(args)