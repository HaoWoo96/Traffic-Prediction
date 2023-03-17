import torch
import logging
import sys

from torchmetrics import Accuracy
# from torchmetrics.functional import precision_recall
from torchmetrics import PrecisionRecallCurve
from tqdm import tqdm

from models import Seq2SeqNoFact, Seq2SeqFact
from data_loader import get_inference_data_loader, get_sorted_inference_data_loader
from utils import create_dir, log_eval_meta, log_eval_spd_result_tmc, log_eval_spd_result_xd, visualize_attn_weight, log_eval_meta, log_lasso_result_tmc, log_lasso_result_xd
from train import create_parser

##################################
#       EVALUATION FUNCTIONS     #
##################################
def eval_error(infer_dataloader, model, factorization, args):
    '''
    FUNCTION 
        Make Inference and Evaluate Prediction Accuracy (for TrafficSeq2Seq & Traffic models)

    INPUTs
        infer_dataloader: dataloader of inference data
        model: a model, either no_fact or finetune
        factorization: 
            bool variable denoting whether the model passed in incorporates factorization or not
        args: miscellaneous arguments
    
    OUTPUTs:
        xxx_root_mse: tensor or dict of tensors in size (seq_len_out), root mean square error for each output time slot
        xxx_mean_ape: tensor or dict of tensors in size (seq_len_out), mean absolute percentage error for each output time slot
        inc_accu, inc_precision, inc_recall: float, evaluatation results of incident status prediction  
        xxx_attn_weights: a tensor or a list of tensors of size (seq_len_out, seq_len_in), average attention weights
    '''
    # initialization
    mapping = {"all":[0, args.dim_out], "truck":[args.dim_out, 2*args.dim_out], "pv":[2*args.dim_out, 3*args.dim_out]}

    # following variables store overall speed prediction errors 
    all_square_error, rec_square_error, nonrec_square_error = 0, 0, 0
    all_abs_perc_error, rec_abs_perc_error, nonrec_abs_perc_error = 0, 0, 0
    instance_cnt, rec_cnt, nonrec_cnt = 0, 0, 0

    inc_predictions = []
    inc_targets = []

    if not factorization:
        attn_weights = 0
    else:
        attn_weights = [0,0,0]

    with torch.no_grad():
        model.eval() # deactivate dropout
        for x, target in tqdm(infer_dataloader):
            # Load Data
            x = x.to(args.device)  # (batch_size, seq_len_in, dim_in)
            target = target.to(args.device)  # (batch_size, seq_len_out + 1, dim_out, 2) for TrafficModel, (batch_size, seq_len_out + 1, dim_out) for TrafficSeq2Seq

            batch_size = x.size(0)
            instance_cnt += batch_size

            inc_target = target[:, 1:, :, -1]
            spd_target = target[:, 1:, :, 0]

            rec_mask = inc_target < 0.5   # (batch_size, seq_len_out, dim_out)
            nonrec_mask = inc_target >= 0.5  # (batch_size, seq_len_out, dim_out)
            inc_targets.append(inc_target)  # list of (batch_size, seq_len_out, dim_out)
            rec_cnt += rec_mask.sum(axis=0).sum(axis=1)  # (seq_len_out,) 
            nonrec_cnt += nonrec_mask.sum(axis=0).sum(axis=1)  # (seq_len_out,)

            # Make Prediction and Update Attention Weights
            if not factorization:
                spd_pred, _, weights = model(x, target) 
                attn_weights += torch.sum(weights, axis=0)  # (seq_len_out, seq_len_in)
            else:
                spd_pred, inc_pred, weights = model(x, target) 
                inc_predictions.append(inc_pred)  # entries are logits, list of (batch_size, seq_len_out, dim_out)
                attn_weights = [attn_weights[i]+torch.sum(weights[i], axis=0) for i in range(3)]

            # Compute Speed Prediction Error
            squre_error_matrix = (spd_pred-spd_target)**2  # for tmc ground truth: (batch_size, args.seq_len_out, args.dim_out*3); for xd ground truth: (batch_size, args.seq_len_out, args.dim_out)
            abs_perc_error_matrix = torch.abs(spd_pred-spd_target)/spd_target  # for tmc ground truth: (batch_size, args.seq_len_out, args.dim_out*3); for xd ground truth: (batch_size, args.seq_len_out, args.dim_out)
            
            all_square_error += torch.sum(squre_error_matrix, axis=0)  # (seq_len_out, dim_out)
            rec_square_error += torch.sum(squre_error_matrix * rec_mask, axis=0)  # (seq_len_out, dim_out)
            nonrec_square_error += torch.sum(squre_error_matrix * nonrec_mask, axis=0)  # (seq_len_out, dim_out)

            all_abs_perc_error += torch.sum(abs_perc_error_matrix, axis=0)  # (seq_len_out, dim_out)
            rec_abs_perc_error += torch.sum(abs_perc_error_matrix * rec_mask, axis=0)  # (seq_len_out, dim_out)
            nonrec_abs_perc_error += torch.sum(abs_perc_error_matrix * nonrec_mask, axis=0)  # (seq_len_out, dim_out)
        
        # Evaluate Speed Prediction (RMSE & MAPE)
        # mean square error
        all_root_mse = (torch.sum(all_square_error, axis=1)/(instance_cnt*args.dim_out))**0.5  # (seq_len_out)
        rec_root_mse = (torch.sum(rec_square_error, axis=1)/rec_cnt)**0.5  # (seq_len_out)
        nonrec_root_mse = (torch.sum(nonrec_square_error, axis=1)/nonrec_cnt)**0.5  # (seq_len_out)

        # mean absolute percentage error
        all_mean_ape = torch.sum(all_abs_perc_error, axis=1)/(instance_cnt*args.dim_out)  # (seq_len_out)
        rec_mean_ape = torch.sum(rec_abs_perc_error, axis=1)/rec_cnt  # (seq_len_out)
        nonrec_mean_ape = torch.sum(nonrec_abs_perc_error, axis=1)/nonrec_cnt  # (seq_len_out)
        
        # (Evaluate Incident Status Prediction &) Compute Average Attention Weights  
        if not factorization:
            attn_weights /= instance_cnt  # (seq_len_out, seq_len_in)
            return all_root_mse, rec_root_mse, nonrec_root_mse, all_mean_ape, rec_mean_ape, nonrec_mean_ape, attn_weights
        else:
            # compute average attention weights
            attn_weights = [a/instance_cnt for a in attn_weights]

            # evaluate incident status prediction
            # accu_metric = Accuracy(args.inc_threshold)  # only works with older version of torchmetrics, i.e. 0.6.0
            accu_metric = Accuracy(task="binary", threshold=args.inc_threshold).to(args.device)
            inc_targets = torch.cat(inc_targets, axis=0).type(torch.int)  # (instance_cnt, seq_len_out, dim_out) 
            inc_predictions = torch.cat(inc_predictions, axis=0)  # (instance_cnt, seq_len_out, dim_out)

            sigm = torch.nn.Sigmoid()
            inc_accu = accu_metric(sigm(inc_predictions), inc_targets) 
            # inc_precision, inc_recall = precision_recall(sigm(inc_predictions), inc_targets)  # only works with older version of torchmetrics, i.e. 0.6.0
            pr_curve = PrecisionRecallCurve(task="binary", thresholds=11).to(args.device)
            inc_precision, inc_recall, inc_thresholds = pr_curve(sigm(inc_predictions), inc_targets)

            return all_root_mse, rec_root_mse, nonrec_root_mse, all_mean_ape, rec_mean_ape, nonrec_mean_ape, attn_weights, inc_accu, inc_precision, inc_recall, inc_thresholds       


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
        xxx_root_mse: tensor or dict of tensor in size (seq_len_out), root mean square error for each output time slot
        xxx_mean_ape: tensor or dict of tensor in size (seq_len_out), mean absolute percentage error for each output time slot
    '''
    # initialization
    mapping = {"all":[0, args.dim_out], "truck":[args.dim_out, 2*args.dim_out], "pv":[2*args.dim_out, 3*args.dim_out]}
    
    # following variables store overall speed prediction errors 
    all_square_error, rec_square_error, nonrec_square_error = 0, 0, 0
    all_abs_perc_error, rec_abs_perc_error, nonrec_abs_perc_error = 0, 0, 0

    instance_cnt, rec_cnt, nonrec_cnt = 0, 0, 0

    with torch.no_grad():
        for _, target in tqdm(infer_dataloader):
            target = target.to(args.device)  # (batch_size, seq_len_out + 1, dim_out, 2) for TrafficModel, (batch_size, seq_len_out + 1, dim_out) for TrafficSeq2Seq

            batch_size = target.size(0)
            instance_cnt += batch_size

            # get target
            inc_target = target[:, 1:, :, -1]
            spd_target = target[:, 1:, :, 0]
            # make Prediction
            spd_pred = target[:, 0, :, 0].unsqueeze(1).repeat(1,args.seq_len_out,1)

            rec_mask = inc_target < 0.5   # (batch_size, seq_len_out, dim_out)
            nonrec_mask = inc_target >= 0.5  # (batch_size, seq_len_out, dim_out)
            rec_cnt += rec_mask.sum(axis=0).sum(axis=1)  # (seq_len_out,) 
            nonrec_cnt += nonrec_mask.sum(axis=0).sum(axis=1)  # (seq_len_out,)

            # Compute Speed Prediction Error
            squre_error_matrix = (spd_pred-spd_target)**2  # for tmc ground truth: (batch_size, args.seq_len_out, args.dim_out*3); for xd ground truth: (batch_size, args.seq_len_out, args.dim_out)
            abs_perc_error_matrix = torch.abs(spd_pred-spd_target)/spd_target  # for tmc ground truth: (batch_size, args.seq_len_out, args.dim_out*3); for xd ground truth: (batch_size, args.seq_len_out, args.dim_out)
            
            all_square_error += torch.sum(squre_error_matrix, axis=0)  # (seq_len_out, dim_out)
            rec_square_error += torch.sum(squre_error_matrix * rec_mask, axis=0)  # (seq_len_out, dim_out)
            nonrec_square_error += torch.sum(squre_error_matrix * nonrec_mask, axis=0)  # (seq_len_out, dim_out)

            all_abs_perc_error += torch.sum(abs_perc_error_matrix, axis=0)  # (seq_len_out, dim_out)
            rec_abs_perc_error += torch.sum(abs_perc_error_matrix * rec_mask, axis=0)  # (seq_len_out, dim_out)
            nonrec_abs_perc_error += torch.sum(abs_perc_error_matrix * nonrec_mask, axis=0)  # (seq_len_out, dim_out)
        
        # Evaluate Speed Prediction (RMSE & MAPE)
        # mean square error
        all_root_mse = (torch.sum(all_square_error, axis=1)/(instance_cnt*args.dim_out))**0.5  # (seq_len_out)
        rec_root_mse = (torch.sum(rec_square_error, axis=1)/rec_cnt)**0.5  # (seq_len_out)
        nonrec_root_mse = (torch.sum(nonrec_square_error, axis=1)/nonrec_cnt)**0.5  # (seq_len_out)

        # mean absolute percentage error
        all_mean_ape = torch.sum(all_abs_perc_error, axis=1)/(instance_cnt*args.dim_out)  # (seq_len_out)
        rec_mean_ape = torch.sum(rec_abs_perc_error, axis=1)/rec_cnt  # (seq_len_out)
        nonrec_mean_ape = torch.sum(nonrec_abs_perc_error, axis=1)/nonrec_cnt  # (seq_len_out)

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
    logging.basicConfig(filename=f"{args.log_dir}/{args.exp_name}/inference_{args.model_type}.log", filemode="w", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG) 


    # 3 Log meta info of model evaluation
    log_eval_meta(args)


    # 4. Initialize Models
    base_model = Seq2SeqNoFact(args).to(args.device)
    traffic_model = Seq2SeqFact(args).to(args.device)
    traffic_model_use_inc_gt = Seq2SeqFact(args).to(args.device)
    traffic_model_use_inc_gt.args.use_inc_gt = 1
    

    # 5. Load Model Checkpoints
    # Base models (seq2seq models) are trained without args.use_expectation, although it doesn't make any difference whether args.use_expectation is true or not
    # Therefore, pretrained base models are all marked with "best_exp_x_x_x_x_x_x_F0.5.pt".
    # args.use_expectation does make a difference in 2-stage model (Traffic model)
    base_model_path = "{}/no_fact/best_{}_{}_F0.5.pt".format(args.checkpoint_dir, args.model_type, "_".join(args.exp_name.split("_")[:-1]))
    traffic_model_path = "{}/finetune/best_{}_{}.pt".format(args.checkpoint_dir, args.model_type, args.exp_name)
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
    

    # 7. Get Evaluation Results
    # result for baseline 1 - Seq2Seq
    base_all_root_mse, base_rec_root_mse, base_nonrec_root_mse, base_all_mean_ape, base_rec_mean_ape, base_nonrec_mean_ape, base_attn_weight = eval_error(infer_dataloader=infer_dataloader, model=base_model, factorization=False, args=args)

    # result for 2-stage model
    traffic_all_root_mse, traffic_rec_root_mse, traffic_nonrec_root_mse, traffic_all_mean_ape, traffic_rec_mean_ape, traffic_nonrec_mean_ape, [traffic_LR_attn_weight, traffic_rec_attn_weight, traffic_nonrec_attn_weight], inc_accu, inc_precision, inc_recall, inc_thresholds = eval_error(infer_dataloader=infer_dataloader, model=traffic_model, factorization=True, args=args)

    # result for 2-stage model assuming perfect incident prediction
    inc_gt_traffic_all_root_mse, inc_gt_traffic_rec_root_mse, inc_gt_traffic_nonrec_root_mse, inc_gt_traffic_all_mean_ape, inc_gt_traffic_rec_mean_ape, inc_gt_traffic_nonrec_mean_ape, _, _, _, _, _ = eval_error(infer_dataloader=infer_dataloader, model=traffic_model_use_inc_gt, factorization=True, args=args)

    # result for baseline 3 - latest observation
    lo_all_root_mse, lo_rec_root_mse, lo_nonrec_root_mse, lo_all_mean_ape, lo_rec_mean_ape, lo_nonrec_mean_ape = eval_last_obs(infer_dataloader=infer_dataloader, args=args)


    # 8. Log Evaluation Results
    logging.info('{:*^100}'.format(" EXPERIMENT RESULT "))
    logging.info('{:=^100}'.format(" 2-Stage Traffic Model "))
    log_eval_spd_result_xd(traffic_all_root_mse, traffic_rec_root_mse, traffic_nonrec_root_mse, traffic_all_mean_ape, traffic_rec_mean_ape, traffic_nonrec_mean_ape)
    # log result for 2-stage model assuming perfect incident prediction
    logging.info(f"Incident Prediction - Accuracy (threshold={args.inc_threshold}):{inc_accu},  Precision:{inc_precision},  Recall:{inc_recall}, PR-Curve Thresholds:{inc_thresholds}\n")
    logging.info('{:-^100}'.format(" assuming perfect incident status prediction "))
    log_eval_spd_result_xd(inc_gt_traffic_all_root_mse, inc_gt_traffic_rec_root_mse, inc_gt_traffic_nonrec_root_mse, inc_gt_traffic_all_mean_ape, inc_gt_traffic_rec_mean_ape, inc_gt_traffic_nonrec_mean_ape)
    logging.info(" ")

    # Logging Baseline Results
    logging.info('{:=^100}'.format(" Baseline 1 - Seq2Seq "))
    log_eval_spd_result_xd(base_all_root_mse, base_rec_root_mse, base_nonrec_root_mse, base_all_mean_ape, base_rec_mean_ape, base_nonrec_mean_ape)
    logging.info(" ")

    logging.info('{:=^100}'.format(" Baseline 2 - LASSO "))
    # TODO
    # log_lasso_result_xd()

    logging.info('{:=^100}'.format(" Baseline 3 - Latest Observations "))
    log_eval_spd_result_xd(lo_all_root_mse, lo_rec_root_mse, lo_nonrec_root_mse, lo_all_mean_ape, lo_rec_mean_ape, lo_nonrec_mean_ape)
    logging.info(" ")

    logging.info('{:=^100}'.format(" Baseline 4 - Historical Average "))
    logging.info(f"RMSE - all: 7.5180")
    logging.info(f"MAPE - all: 0.1827")

    # 9. Visualize Attention Weights
    logging.info(f"Please check visualizations of attention weights under folder {args.log_dir}/{args.exp_name}")

    # base model
    base_attn_weight_path = f"{args.log_dir}/{args.exp_name}/{args.model_type}_no_fact_attn_weight.jpg"
    base_attn_weight_title = f"Attention Weight of Seq2Seq {args.exp_name}"
    visualize_attn_weight(base_attn_weight.cpu(), args, base_attn_weight_title, base_attn_weight_path)

    # LR/Rec/Nonrec decoders of 2-stage model
    traffic_LR_attn_weight_path = f"{args.log_dir}/{args.exp_name}/{args.model_type}_LR_attn_weight.jpg"
    traffic_LR_attn_weight_title = f"Attention Weight of LR Module in Traffic Model {args.exp_name}"
    visualize_attn_weight(traffic_LR_attn_weight.cpu(), args, traffic_LR_attn_weight_title, traffic_LR_attn_weight_path)

    traffic_rec_attn_weight_path = f"{args.log_dir}/{args.exp_name}/{args.model_type}_rec_attn_weight.jpg"
    traffic_rec_attn_weight_title = f"Attention Weight of Recurrent Decoder in Traffic Model {args.exp_name}"
    visualize_attn_weight(traffic_rec_attn_weight.cpu(), args, traffic_rec_attn_weight_title, traffic_rec_attn_weight_path)

    traffic_nonrec_attn_weight_path = f"{args.log_dir}/{args.exp_name}/{args.model_type}_nonrec_attn_weight.jpg"
    traffic_nonrec_attn_weight_title = f"Attention Weight of Nonrecurrent Decoder in Traffic Model {args.exp_name}"
    visualize_attn_weight(traffic_nonrec_attn_weight.cpu(), args, traffic_nonrec_attn_weight_title, traffic_nonrec_attn_weight_path)

    # 2-stage model
    traffic_spd_attn_weight = (traffic_rec_attn_weight + traffic_nonrec_attn_weight)/2
    traffic_spd_attn_weight_path = f"{args.log_dir}/{args.exp_name}/{args.model_type}_spd_attn_weight.jpg"
    traffic_spd_attn_weight_title = f"Attention Weight of Speed Prediction in Traffic Model {args.exp_name}"
    visualize_attn_weight(traffic_spd_attn_weight.cpu(), args, traffic_spd_attn_weight_title, traffic_spd_attn_weight_path)
    

if __name__ == "__main__":

    # 1. Modify Arguments
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # For reproducibility
    torch.manual_seed(args.seed)

    # Task specific directories
    args.log_dir = "./results" 

    args.exp_name = args.model_type
    args.exp_name = f"{str(args.use_dens)[0]}_{str(args.use_spd_all)[0]}_{str(args.use_spd_truck)[0]}_{str(args.use_spd_pv)[0]}_{args.seq_len_in}_{args.seq_len_out}_{args.freq_out}_{str(args.use_expectation)[0]}"
    if not args.use_expectation:
        args.exp_name += str(args.inc_threshold)

    # Change input dimension based on task type and whether to use new features or not
    if not args.use_dens:
        args.dim_in -= 207 
    if not args.use_spd_all:
        args.dim_in -= 207
    if not args.use_spd_truck:
        args.dim_in -= 207
    if not args.use_spd_pv:
        args.dim_in -= 207
    
    # 2. Execute Inference Pipeline
    main(args)
