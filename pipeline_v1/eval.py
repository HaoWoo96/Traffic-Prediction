import torch
import logging
import sys

from torchmetrics import Accuracy
from torchmetrics.functional import precision_recall
from tqdm import tqdm

from models import TrafficSeq2Seq, TrafficModel
from pipeline_v2.data_loader import get_inference_data_loader, get_sorted_inference_data_loader
from utils import create_dir, log_eval_meta, log_eval_spd_result_tmc, log_eval_spd_result_xd, visualize_attn_weight, log_eval_meta, log_lasso_result_tmc, log_lasso_result_xd
from train import create_parser

##################################
#       EVALUATION FUNCTIONS     #
##################################
def eval_error(infer_dataloader, model, model_type, args):
    '''
    FUNCTION 
        Make Inference and Evaluate Prediction Accuracy (for TrafficSeq2Seq & Traffic models)

    INPUTs
        infer_dataloader: dataloader of inference data
        model: basic seq2seq model (TrafficSeq2Seq) or finetuned 2-stage model (Traffic)
        model_type: "base" or "traffic"
        args: miscellaneous arguments
    
    OUTPUTs:
        xxx_root_mse: tensor or dict of tensors in size (out_seq_len), root mean square error for each output time slot
        xxx_mean_ape: tensor or dict of tensors in size (out_seq_len), mean absolute percentage error for each output time slot
        inc_accu, inc_precision, inc_recall: float, evaluatation results of incident status prediction  
        xxx_attn_weights: a tensor or a list of tensors of size (out_seq_len, in_seq_len), average attention weights
    '''
    # initialization
    mapping = {"all":[0, args.out_dim], "truck":[args.out_dim, 2*args.out_dim], "pv":[2*args.out_dim, 3*args.out_dim]}
    if args.gt_type == "tmc":
        # following dictionaries store speed prediction errors for all vehicles / trucks / personal vehicles
        all_square_error, rec_square_error, nonrec_square_error = {}, {}, {}
        all_abs_perc_error, rec_abs_perc_error, nonrec_abs_perc_error = {}, {}, {}
        for s in mapping.keys():
            all_square_error[s] = 0
            rec_square_error[s] = 0
            nonrec_square_error[s] = 0
            all_abs_perc_error[s] = 0
            rec_abs_perc_error[s] = 0
            nonrec_abs_perc_error[s] = 0
    else:
        # following variables store overall speed prediction errors 
        all_square_error, rec_square_error, nonrec_square_error = 0, 0, 0
        all_abs_perc_error, rec_abs_perc_error, nonrec_abs_perc_error = 0, 0, 0
    instance_cnt, rec_cnt, nonrec_cnt = 0, 0, 0

    inc_predictions = []
    inc_targets = []

    if model_type == "base":
        attn_weights = 0
    else:
        attn_weights = [0,0,0]

    with torch.no_grad():
        model.eval() # deactivate dropout
        for x, target in tqdm(infer_dataloader):
            # Load Data
            x = x.to(args.device)  # (batch_size, in_seq_len, in_dim)
            target = target.to(args.device)  # (batch_size, out_seq_len + 1, out_dim, 2) for TrafficModel, (batch_size, out_seq_len + 1, out_dim) for TrafficSeq2Seq

            batch_size = x.size(0)
            instance_cnt += batch_size

            if args.gt_type == "tmc":
                inc_target = target[:, 1:, :, 3]
                spd_target = target[:, 1:, :, :3].reshape(batch_size, args.out_seq_len, args.out_dim*3)
            else:
                inc_target = target[:, 1:, :, 1]
                spd_target = target[:, 1:, :, 0]
            rec_mask = inc_target < 0.5   # (batch_size, out_seq_len, out_dim)
            nonrec_mask = inc_target >= 0.5  # (batch_size, out_seq_len, out_dim)
            inc_targets.append(inc_target)  # list of (batch_size, out_seq_len, out_dim)
            rec_cnt += rec_mask.sum(axis=0).sum(axis=1)  # (out_seq_len,) 
            nonrec_cnt += nonrec_mask.sum(axis=0).sum(axis=1)  # (out_seq_len,)

            # Make Prediction and Update Attention Weights
            if model_type == "base":
                spd_pred, _, weights = model(x, target) 
                attn_weights += torch.sum(weights, axis=0)  # (out_seq_len, in_seq_len)
            else:
                spd_pred, inc_pred, weights = model(x, target) 
                inc_predictions.append(inc_pred)  # entries are logits, list of (batch_size, out_seq_len, out_dim)
                attn_weights = [attn_weights[i]+torch.sum(weights[i], axis=0) for i in range(3)]

            # Compute Speed Prediction Error
            squre_error_matrix = (spd_pred-spd_target)**2  # for tmc ground truth: (batch_size, args.out_seq_len, args.out_dim*3); for xd ground truth: (batch_size, args.out_seq_len, args.out_dim)
            abs_perc_error_matrix = torch.abs(spd_pred-spd_target)/spd_target  # for tmc ground truth: (batch_size, args.out_seq_len, args.out_dim*3); for xd ground truth: (batch_size, args.out_seq_len, args.out_dim)
            if args.gt_type == "tmc":
                for s, [i, j] in mapping.items():
                    # square error
                    all_square_error[s] += torch.sum(squre_error_matrix[:, :, i:j], axis=0)  # (out_seq_len, out_dim)
                    rec_square_error[s] += torch.sum(squre_error_matrix[:, :, i:j] * rec_mask, axis=0)  # (out_seq_len, out_dim)
                    nonrec_square_error[s] += torch.sum(squre_error_matrix[:, :, i:j] * nonrec_mask, axis=0)  # (out_seq_len, out_dim)

                    # absolute percentage error
                    all_abs_perc_error[s] += torch.sum(abs_perc_error_matrix[:, :, i:j], axis=0)  # (out_seq_len, out_dim)
                    rec_abs_perc_error[s] += torch.sum(abs_perc_error_matrix[:, :, i:j] * rec_mask, axis=0)  # (out_seq_len, out_dim)
                    nonrec_abs_perc_error[s] += torch.sum(abs_perc_error_matrix[:, :, i:j] * nonrec_mask, axis=0)  # (out_seq_len, out_dim)
            else:
                all_square_error += torch.sum(squre_error_matrix, axis=0)  # (out_seq_len, out_dim)
                rec_square_error += torch.sum(squre_error_matrix * rec_mask, axis=0)  # (out_seq_len, out_dim)
                nonrec_square_error += torch.sum(squre_error_matrix * nonrec_mask, axis=0)  # (out_seq_len, out_dim)

                all_abs_perc_error += torch.sum(abs_perc_error_matrix, axis=0)  # (out_seq_len, out_dim)
                rec_abs_perc_error += torch.sum(abs_perc_error_matrix * rec_mask, axis=0)  # (out_seq_len, out_dim)
                nonrec_abs_perc_error += torch.sum(abs_perc_error_matrix * nonrec_mask, axis=0)  # (out_seq_len, out_dim)
        
        # Evaluate Speed Prediction (RMSE & MAPE)
        if args.gt_type == "tmc":
            all_root_mse, rec_root_mse, nonrec_root_mse, all_mean_ape, rec_mean_ape, nonrec_mean_ape = {}, {}, {}, {}, {}, {}
            for s in mapping.keys():
                # mean square error
                all_root_mse[s] = (torch.sum(all_square_error[s], axis=1)/(instance_cnt*args.out_dim))**0.5  # (out_seq_len)
                rec_root_mse[s] = (torch.sum(rec_square_error[s], axis=1)/rec_cnt)**0.5  # (out_seq_len)
                nonrec_root_mse[s] = (torch.sum(nonrec_square_error[s], axis=1)/nonrec_cnt)**0.5  # (out_seq_len)

                # mean absolute percentage error
                all_mean_ape[s] = torch.sum(all_abs_perc_error[s], axis=1)/(instance_cnt*args.out_dim)  # (out_seq_len)
                rec_mean_ape[s] = torch.sum(rec_abs_perc_error[s], axis=1)/rec_cnt  # (out_seq_len)
                nonrec_mean_ape[s] = torch.sum(nonrec_abs_perc_error[s], axis=1)/nonrec_cnt  # (out_seq_len)
        else:
            # mean square error
            all_root_mse = (torch.sum(all_square_error, axis=1)/(instance_cnt*args.out_dim))**0.5  # (out_seq_len)
            rec_root_mse = (torch.sum(rec_square_error, axis=1)/rec_cnt)**0.5  # (out_seq_len)
            nonrec_root_mse = (torch.sum(nonrec_square_error, axis=1)/nonrec_cnt)**0.5  # (out_seq_len)

            # mean absolute percentage error
            all_mean_ape = torch.sum(all_abs_perc_error, axis=1)/(instance_cnt*args.out_dim)  # (out_seq_len)
            rec_mean_ape = torch.sum(rec_abs_perc_error, axis=1)/rec_cnt  # (out_seq_len)
            nonrec_mean_ape = torch.sum(nonrec_abs_perc_error, axis=1)/nonrec_cnt  # (out_seq_len)
        
        # (Evaluate Incident Status Prediction &) Compute Average Attention Weights  
        if model_type == "base":
            attn_weights /= instance_cnt  # (out_seq_len, in_seq_len)
            return all_root_mse, rec_root_mse, nonrec_root_mse, all_mean_ape, rec_mean_ape, nonrec_mean_ape, attn_weights
        else:
            # compute average attention weights
            attn_weights = [a/instance_cnt for a in attn_weights]

            # evaluate incident status prediction
            accu_metric = Accuracy(args.inc_threshold)
            inc_targets = torch.cat(inc_targets, axis=0).type(torch.int)  # (instance_cnt, out_seq_len, out_dim) 
            inc_predictions = torch.cat(inc_predictions, axis=0)  # (instance_cnt, out_seq_len, out_dim)

            sigm = torch.nn.Sigmoid()
            inc_accu = accu_metric(sigm(inc_predictions), inc_targets) 
            inc_precision, inc_recall = precision_recall(sigm(inc_predictions), inc_targets)

            return all_root_mse, rec_root_mse, nonrec_root_mse, all_mean_ape, rec_mean_ape, nonrec_mean_ape, attn_weights, inc_accu, inc_precision, inc_recall       


def eval_timeliness(sorted_infer_dataloader, model, args):
    '''
    FUNCTION
        For Traffic model, make inference on data sorted by time
        Evaluate on timeliness of incident status prediction
    
    OUTPUT
        avg_time_ahead: average number of mins ahead achieved by model incident prediction compared to Waze/RCRS report
    '''
    # compute average time ahead of waze/RCRS report
    pass


def eval_last_obs(infer_dataloader, args):
    '''
    FUNCTION 
        One of the Baseline Methods
        Make Inference Based on Latest Observation and Compute Loss

    INPUTs
        infer_dataloader: dataloader of inference data
        args: arguments
    
    OUTPUTs:
        xxx_root_mse: tensor or dict of tensor in size (out_seq_len), root mean square error for each output time slot
        xxx_mean_ape: tensor or dict of tensor in size (out_seq_len), mean absolute percentage error for each output time slot
    '''
    # initialization
    mapping = {"all":[0, args.out_dim], "truck":[args.out_dim, 2*args.out_dim], "pv":[2*args.out_dim, 3*args.out_dim]}
    if args.gt_type == "tmc":
        # following dictionaries store speed prediction errors for all vehicles / trucks / personal vehicles
        all_square_error, rec_square_error, nonrec_square_error = {}, {}, {}
        all_abs_perc_error, rec_abs_perc_error, nonrec_abs_perc_error = {}, {}, {}
        for s in mapping.keys():
            all_square_error[s] = 0
            rec_square_error[s] = 0
            nonrec_square_error[s] = 0
            all_abs_perc_error[s] = 0
            rec_abs_perc_error[s] = 0
            nonrec_abs_perc_error[s] = 0
    else:
        # following variables store overall speed prediction errors 
        all_square_error, rec_square_error, nonrec_square_error = 0, 0, 0
        all_abs_perc_error, rec_abs_perc_error, nonrec_abs_perc_error = 0, 0, 0
    instance_cnt, rec_cnt, nonrec_cnt = 0, 0, 0

    with torch.no_grad():
        for _, target in tqdm(infer_dataloader):
            target = target.to(args.device)  # (batch_size, out_seq_len + 1, out_dim, 2) for TrafficModel, (batch_size, out_seq_len + 1, out_dim) for TrafficSeq2Seq

            batch_size = target.size(0)
            instance_cnt += batch_size

            if args.gt_type == "tmc":
                # get target
                inc_target = target[:, 1:, :, 3]
                spd_target = target[:, 1:, :, :3].reshape(batch_size, args.out_seq_len, args.out_dim*3)
                # make prediction
                spd_pred = target[:, 0, :, :3].unsqueeze(1).repeat(1,args.out_seq_len,1, 1).reshape(batch_size, args.out_seq_len, args.out_dim*3)
            else:
                # get target
                inc_target = target[:, 1:, :, 1]
                spd_target = target[:, 1:, :, 0]
                # make Prediction
                spd_pred = target[:, 0, :, 0].unsqueeze(1).repeat(1,args.out_seq_len,1)

            rec_mask = inc_target < 0.5   # (batch_size, out_seq_len, out_dim)
            nonrec_mask = inc_target >= 0.5  # (batch_size, out_seq_len, out_dim)
            rec_cnt += rec_mask.sum(axis=0).sum(axis=1)  # (out_seq_len,) 
            nonrec_cnt += nonrec_mask.sum(axis=0).sum(axis=1)  # (out_seq_len,)

            # Compute Speed Prediction Error
            squre_error_matrix = (spd_pred-spd_target)**2  # for tmc ground truth: (batch_size, args.out_seq_len, args.out_dim*3); for xd ground truth: (batch_size, args.out_seq_len, args.out_dim)
            abs_perc_error_matrix = torch.abs(spd_pred-spd_target)/spd_target  # for tmc ground truth: (batch_size, args.out_seq_len, args.out_dim*3); for xd ground truth: (batch_size, args.out_seq_len, args.out_dim)
            if args.gt_type == "tmc":
                for s, [i, j] in mapping.items():
                    # square error
                    all_square_error[s] += torch.sum(squre_error_matrix[:, :, i:j], axis=0)  # (out_seq_len, out_dim)
                    rec_square_error[s] += torch.sum(squre_error_matrix[:, :, i:j] * rec_mask, axis=0)  # (out_seq_len, out_dim)
                    nonrec_square_error[s] += torch.sum(squre_error_matrix[:, :, i:j] * nonrec_mask, axis=0)  # (out_seq_len, out_dim)

                    # absolute percentage error
                    all_abs_perc_error[s] += torch.sum(abs_perc_error_matrix[:, :, i:j], axis=0)  # (out_seq_len, out_dim)
                    rec_abs_perc_error[s] += torch.sum(abs_perc_error_matrix[:, :, i:j] * rec_mask, axis=0)  # (out_seq_len, out_dim)
                    nonrec_abs_perc_error[s] += torch.sum(abs_perc_error_matrix[:, :, i:j] * nonrec_mask, axis=0)  # (out_seq_len, out_dim)
            else:
                all_square_error += torch.sum(squre_error_matrix, axis=0)  # (out_seq_len, out_dim)
                rec_square_error += torch.sum(squre_error_matrix * rec_mask, axis=0)  # (out_seq_len, out_dim)
                nonrec_square_error += torch.sum(squre_error_matrix * nonrec_mask, axis=0)  # (out_seq_len, out_dim)

                all_abs_perc_error += torch.sum(abs_perc_error_matrix, axis=0)  # (out_seq_len, out_dim)
                rec_abs_perc_error += torch.sum(abs_perc_error_matrix * rec_mask, axis=0)  # (out_seq_len, out_dim)
                nonrec_abs_perc_error += torch.sum(abs_perc_error_matrix * nonrec_mask, axis=0)  # (out_seq_len, out_dim)
        
        # Evaluate Speed Prediction (RMSE & MAPE)
        if args.gt_type == "tmc":
            all_root_mse, rec_root_mse, nonrec_root_mse, all_mean_ape, rec_mean_ape, nonrec_mean_ape = {}, {}, {}, {}, {}, {}
            for s in mapping.keys():
                # mean square error
                all_root_mse[s] = (torch.sum(all_square_error[s], axis=1)/(instance_cnt*args.out_dim))**0.5  # (out_seq_len)
                rec_root_mse[s] = (torch.sum(rec_square_error[s], axis=1)/(rec_cnt))**0.5  # (out_seq_len)
                nonrec_root_mse[s] = (torch.sum(nonrec_square_error[s], axis=1)/(nonrec_cnt))**0.5  # (out_seq_len)

                # mean absolute percentage error
                all_mean_ape[s] = torch.sum(all_abs_perc_error[s], axis=1)/(instance_cnt*args.out_dim)  # (out_seq_len)
                rec_mean_ape[s] = torch.sum(rec_abs_perc_error[s], axis=1)/rec_cnt  # (out_seq_len)
                nonrec_mean_ape[s] = torch.sum(nonrec_abs_perc_error[s], axis=1)/nonrec_cnt  # (out_seq_len)
        else:
            # mean square error
            all_root_mse = (torch.sum(all_square_error, axis=1)/(instance_cnt*args.out_dim))**0.5  # (out_seq_len)
            rec_root_mse = (torch.sum(rec_square_error, axis=1)/rec_cnt)**0.5  # (out_seq_len)
            nonrec_root_mse = (torch.sum(nonrec_square_error, axis=1)/nonrec_cnt)**0.5  # (out_seq_len)

            # mean absolute percentage error
            all_mean_ape = torch.sum(all_abs_perc_error, axis=1)/(instance_cnt*args.out_dim)  # (out_seq_len)
            rec_mean_ape = torch.sum(rec_abs_perc_error, axis=1)/rec_cnt  # (out_seq_len)
            nonrec_mean_ape = torch.sum(nonrec_abs_perc_error, axis=1)/nonrec_cnt  # (out_seq_len)

    return all_root_mse, rec_root_mse, nonrec_root_mse, all_mean_ape, rec_mean_ape, nonrec_mean_ape


##################################
#       EVALUATION PIPELINE     #
##################################
def main(args):
    '''
    Load Model, Make Inference and Visualize Results if Applicable
    '''
    # 1. Create Directories
    create_dir(args.log_dir)
    create_dir(f"{args.log_dir}/{args.exp_name}")

    # 2. Set up Logger
    logging.basicConfig(filename=f"{args.log_dir}/{args.exp_name}/inference.log", filemode="w", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG) 

    # 3 Log meta info of model evaluation
    log_eval_meta(args)

    # 4. Initialize Models
    base_model = TrafficSeq2Seq(args)
    traffic_model = TrafficModel(args)
    traffic_model_use_inc_gt = TrafficModel(args)
    traffic_model_use_inc_gt.args.use_inc_gt = 1
    
    # 5. Load Model Checkpoints
    # Base models (seq2seq models) are trained with args.use_expectation, although it doesn't make any difference whether args.use_expectation is true or not
    # Therefore, pretrained base models are all marked with "best_exp_x_x_x_x_x_x_T.pt".
    # args.use_expectation does make a difference in 2-stage model (Traffic model)
    base_model_path = "{}/base/best_{}_T.pt".format(args.checkpoint_dir, "_".join(args.exp_name.split("_")[:-1]))
    traffic_model_path = "{}/finetune/best_{}.pt".format(args.checkpoint_dir, args.exp_name)
    with open(base_model_path, "rb") as f_base_model, open(traffic_model_path, "rb") as f_traffic_model:
        # load state dict
        state_dict_base = torch.load(f_base_model, map_location=args.device)
        state_dict_traffic = torch.load(f_traffic_model, map_location=args.device)

        # populate state dict into models and log
        base_model.load_state_dict(state_dict_base)
        logging.info(f"successfully loaded checkpoint from {base_model_path}")

        traffic_model.load_state_dict(state_dict_traffic)
        traffic_model_use_inc_gt.load_state_dict(state_dict_traffic)
        logging.info(f"successfully loaded checkpoint from {traffic_model_path}")
    
    # 6. Load Data for Inference
    infer_dataloader = get_inference_data_loader(args)
    # sorted_infer_dataloader = get_sorted_inference_data_loader(args)
    logging.info(f"successfully loaded data \n")

    # 7. Get and Log Evaluation Results
    logging.info('{:*^100}'.format(" EXPERIMENT RESULT "))
    
    # result for baseline 1 - Seq2Seq
    base_all_root_mse, base_rec_root_mse, base_nonrec_root_mse, base_all_mean_ape, base_rec_mean_ape, base_nonrec_mean_ape, base_attn_weight = eval_error(infer_dataloader=infer_dataloader, model=base_model, model_type="base", args=args)
    # result for 2-stage model
    traffic_all_root_mse, traffic_rec_root_mse, traffic_nonrec_root_mse, traffic_all_mean_ape, traffic_rec_mean_ape, traffic_nonrec_mean_ape, [traffic_LR_attn_weight, traffic_rec_attn_weight, traffic_nonrec_attn_weight], inc_accu, inc_precision, inc_recall = eval_error(infer_dataloader=infer_dataloader, model=traffic_model, model_type="traffic", args=args)
    # result for 2-stage model assuming perfect incident prediction
    inc_gt_traffic_all_root_mse, inc_gt_traffic_rec_root_mse, inc_gt_traffic_nonrec_root_mse, inc_gt_traffic_all_mean_ape, inc_gt_traffic_rec_mean_ape, inc_gt_traffic_nonrec_mean_ape, _, _, _, _ = eval_error(infer_dataloader=infer_dataloader, model=traffic_model_use_inc_gt, model_type="traffic", args=args)
    # result for baseline 3 - latest observation
    lo_all_root_mse, lo_rec_root_mse, lo_nonrec_root_mse, lo_all_mean_ape, lo_rec_mean_ape, lo_nonrec_mean_ape = eval_last_obs(infer_dataloader=infer_dataloader, args=args)

    if args.gt_type == "tmc":
        logging.info('{:=^100}'.format(" 2-Stage Traffic Model "))
        log_eval_spd_result_tmc(traffic_all_root_mse, traffic_rec_root_mse, traffic_nonrec_root_mse, traffic_all_mean_ape, traffic_rec_mean_ape, traffic_nonrec_mean_ape)
        logging.info(f"Incident Prediction - Accuracy:{inc_accu},  Precision:{inc_precision},  Recall:{inc_recall}\n")
        # log result for 2-stage model assuming perfect incident prediction
        logging.info('{:-^100}'.format(" assuming perfect incident status prediction "))
        log_eval_spd_result_tmc(inc_gt_traffic_all_root_mse, inc_gt_traffic_rec_root_mse, inc_gt_traffic_nonrec_root_mse, inc_gt_traffic_all_mean_ape, inc_gt_traffic_rec_mean_ape, inc_gt_traffic_nonrec_mean_ape)
        logging.info(" ")

        logging.info('{:=^100}'.format(" Baseline 1 - Seq2Seq "))
        log_eval_spd_result_tmc(base_all_root_mse, base_rec_root_mse, base_nonrec_root_mse, base_all_mean_ape, base_rec_mean_ape, base_nonrec_mean_ape)

        logging.info('{:=^100}'.format(" Baseline 2 - LASSO "))
        # log_lasso_result_tmc()

        logging.info('{:=^100}'.format(" Baseline 3 - Latest Observations "))
        log_eval_spd_result_tmc(lo_all_root_mse, lo_rec_root_mse, lo_nonrec_root_mse, lo_all_mean_ape, lo_rec_mean_ape, lo_nonrec_mean_ape)

        logging.info('{:=^100}'.format(" Baseline 4 - Historical Average "))
        # TO DO
    else:
        logging.info('{:=^100}'.format(" 2-Stage Traffic Model "))
        log_eval_spd_result_xd(traffic_all_root_mse, traffic_rec_root_mse, traffic_nonrec_root_mse, traffic_all_mean_ape, traffic_rec_mean_ape, traffic_nonrec_mean_ape)
        # log result for 2-stage model assuming perfect incident prediction
        logging.info(f"Incident Prediction - Accuracy:{inc_accu},  Precision:{inc_precision},  Recall:{inc_recall}\n")
        logging.info('{:-^100}'.format(" assuming perfect incident status prediction "))
        log_eval_spd_result_xd(inc_gt_traffic_all_root_mse, inc_gt_traffic_rec_root_mse, inc_gt_traffic_nonrec_root_mse, inc_gt_traffic_all_mean_ape, inc_gt_traffic_rec_mean_ape, inc_gt_traffic_nonrec_mean_ape)
        logging.info(" ")

        logging.info('{:=^100}'.format(" Baseline 1 - Seq2Seq "))
        log_eval_spd_result_xd(base_all_root_mse, base_rec_root_mse, base_nonrec_root_mse, base_all_mean_ape, base_rec_mean_ape, base_nonrec_mean_ape)
        logging.info(" ")

        logging.info('{:=^100}'.format(" Baseline 2 - LASSO "))
        # log_lasso_result_xd()

        logging.info('{:=^100}'.format(" Baseline 3 - Latest Observations "))
        log_eval_spd_result_xd(lo_all_root_mse, lo_rec_root_mse, lo_nonrec_root_mse, lo_all_mean_ape, lo_rec_mean_ape, lo_nonrec_mean_ape)
        logging.info(" ")

        logging.info('{:=^100}'.format(" Baseline 4 - Historical Average "))
        # TO DO

    # 8. Visualize Attention Weights
    logging.info(f"Please check visualizations of attention weights under folder {args.log_dir}/{args.exp_name}")

    # base model
    base_attn_weight_path = f"{args.log_dir}/{args.exp_name}/base_attn_weight.jpg"
    base_attn_weight_title = f"Attention Weight of Seq2Seq {args.exp_name}"
    visualize_attn_weight(base_attn_weight, args, base_attn_weight_title, base_attn_weight_path)

    # LR/Rec/Nonrec decoders of 2-stage model
    traffic_LR_attn_weight_path = f"{args.log_dir}/{args.exp_name}/traffic_LR_attn_weight.jpg"
    traffic_LR_attn_weight_title = f"Attention Weight of LR Module in Traffic Model {args.exp_name}"
    visualize_attn_weight(traffic_LR_attn_weight, args, traffic_LR_attn_weight_title, traffic_LR_attn_weight_path)

    traffic_rec_attn_weight_path = f"{args.log_dir}/{args.exp_name}/traffic_rec_attn_weight.jpg"
    traffic_rec_attn_weight_title = f"Attention Weight of Recurrent Decoder in Traffic Model {args.exp_name}"
    visualize_attn_weight(traffic_rec_attn_weight, args, traffic_rec_attn_weight_title, traffic_rec_attn_weight_path)

    traffic_nonrec_attn_weight_path = f"{args.log_dir}/{args.exp_name}/traffic_nonrec_attn_weight.jpg"
    traffic_nonrec_attn_weight_title = f"Attention Weight of Nonrecurrent Decoder in Traffic Model {args.exp_name}"
    visualize_attn_weight(traffic_nonrec_attn_weight, args, traffic_nonrec_attn_weight_title, traffic_nonrec_attn_weight_path)

    # 2-stage model
    traffic_spd_attn_weight = (traffic_rec_attn_weight + traffic_nonrec_attn_weight)/2
    traffic_spd_attn_weight_path = f"{args.log_dir}/{args.exp_name}/traffic_spd_attn_weight.jpg"
    traffic_spd_attn_weight_title = f"Attention Weight of Speed Prediction in Traffic Model {args.exp_name}"
    visualize_attn_weight(traffic_spd_attn_weight, args, traffic_spd_attn_weight_title, traffic_spd_attn_weight_path)
    

if __name__ == "__main__":

    # 1. Modify Arguments
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # For reproducibility
    torch.manual_seed(args.seed)

    # Task specific directories
    args.log_dir = "./results" 
    args.exp_name += f"_{args.gt_type}_{str(args.use_density)[0]}_{str(args.use_truck_spd)[0]}_{str(args.use_pv_spd)[0]}_{args.in_seq_len}_{args.out_seq_len}_{args.out_freq}_{str(args.use_expectation)[0]}" 

    # Change input dimension based on task type and whether to use new features or not
    if not args.use_density:
        args.in_dim -= 313 
    if not args.use_speed:
        args.in_dim -= 313
    if not args.use_truck_spd:
        args.in_dim -= 233
    if not args.use_pv_spd:
        args.in_dim -= 233
    
    # 2. Execute Inference Pipeline
    main(args)
