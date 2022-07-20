import torch
import logging
import sys

from torchmetrics import Accuracy
from torchmetrics.functional import precision_recall
from tqdm import tqdm

from models import TrafficSeq2Seq, TrafficModel
from data_loader import get_inference_data_loader
from utils import create_dir, visualize_attn_weight
from train import create_parser


def infer(infer_dataloader, model, model_type, args):
    '''
    FUNCTION 
        Make Inference and Compute Loss

    INPUTs
        infer_dataloader: dataloader of inference data
        model: basic seq2seq model (TrafficSeq2Seq) or finetuned Traffic model
        model_type: "base" or "traffic"
        args: miscellaneous arguments
    
    OUTPUTs:
        root_mse: size (out_seq_len), root mean square error for each output time slot
        mean_ape: size (out_seq_len), mean absolute percentage error for each output time slot
        attn_weights: a tensor or a list of tensors of size (out_seq_len, in_seq_len), average attention weights
    '''
    all_square_error, rec_square_error, nonrec_square_error = 0, 0, 0
    all_abs_perc_error, rec_abs_perc_error, nonrec_abs_perc_error = 0, 0, 0
    instance_cnt, rec_cnt, nonrec_cnt = 0, 0, 0

    inc_prediction = []
    inc_target = []

    if model_type == "base":
        attn_weights = 0
    else:
        attn_weights = [0,0,0]

    with torch.no_grad():
        model.eval()
        for i, batch in tqdm(enumerate(infer_dataloader)):
            x, target, recurrent = batch
            x = x.to(args.device)  # (batch_size, in_seq_len, in_dim)
            target = target.to(args.device)  # (batch_size, out_seq_len + 1, out_dim, 2) for TrafficModel, (batch_size, out_seq_len + 1, out_dim) for TrafficSeq2Seq
            recurrent = recurrent.to(args.device)  # (batch_size) 

            instance_cnt += x.size(0)
            rec_cnt += torch.sum(recurrent)
            nonrec_cnt += instance_cnt - rec_cnt

            inc_target.append(target[:, 1:, :, 1])  # (batch_size, out_seq_len, out_dim)

            # Make Prediction and Update Attention Weights
            if model_type == "base":
                spd_pred, weights = model(x, target) 
                attn_weights += torch.sum(weights, axis=0)  # (out_seq_len, in_seq_len)
            else:
                spd_pred, inc_pred, weights = model(x, target) 
                inc_prediction.append(inc_pred)  # logits, (batch_size, out_seq_len, out_dim)
                attn_weights = [attn_weights[i]+torch.sum(weights[i], axis=0) for i in range(3)]

            # Compute Intermediate Error
            all_square_error += torch.sum((spd_pred-target[:, 1:, :, 0])**2, axis=0)  # (out_seq_len, out_dim)
            rec_square_error += torch.sum(((spd_pred-target[:, 1:, :, 0])**2)[recurrent==1], axis=0)  # (out_seq_len, out_dim)
            nonrec_square_error += torch.sum(((spd_pred-target[:, 1:, :, 0])**2)[recurrent==0], axis=0)  # (out_seq_len, out_dim)

            all_abs_perc_error += torch.sum(torch.abs(spd_pred-target[:, 1:, :, 0])/target[:, 1:, :, 0], axis=0)  # (out_seq_len, out_dim)
            rec_abs_perc_error += torch.sum(torch.abs(spd_pred[recurrent==1]-target[:, 1:, :, 0][recurrent==1])/target[:, 1:, :, 0][recurrent==1], axis=0)  # (out_seq_len, out_dim)
            nonrec_abs_perc_error += torch.sum(torch.abs(spd_pred[recurrent==0]-target[:, 1:, :, 0][recurrent==0])/target[:, 1:, :, 0][recurrent==0], axis=0)  # (out_seq_len, out_dim)
        
        # Evaluate Speed Prediction (and Incident Prediction), and Computer Average Attention Weight
        all_root_mse = (torch.sum(all_square_error, axis=1)/(instance_cnt*args.out_dim))**0.5  # (out_seq_len)
        rec_root_mse = (torch.sum(rec_square_error, axis=1)/(rec_cnt*args.out_dim))**0.5  # (out_seq_len)
        nonrec_root_mse = (torch.sum(nonrec_square_error, axis=1)/(nonrec_cnt*args.out_dim))**0.5  # (out_seq_len)

        all_mean_ape = torch.sum(all_abs_perc_error, axis=1)/(instance_cnt*args.out_dim)  # (out_seq_len)
        rec_mean_ape = torch.sum(rec_abs_perc_error, axis=1)/(rec_cnt*args.out_dim)  # (out_seq_len)
        nonrec_mean_ape = torch.sum(nonrec_abs_perc_error, axis=1)/(nonrec_cnt*args.out_dim)  # (out_seq_len)
        
        if model_type == "base":
            attn_weights /= instance_cnt  # (out_seq_len, in_seq_len)

            return all_root_mse, rec_root_mse, nonrec_root_mse, all_mean_ape, rec_mean_ape, nonrec_mean_ape, attn_weights

        else:
            # compute average attention weights
            attn_weights = [a/instance_cnt for a in attn_weights]

            # evaluate incident prediction
            accu_metric = Accuracy(0.5)
            inc_target = torch.cat(inc_target, axis=0).type(torch.int)
            inc_prediction = torch.cat(inc_prediction, axis=0)
            inc_accu = accu_metric(inc_prediction, inc_target)

            inc_precision, inc_recall = precision_recall(inc_prediction, inc_target)

            return all_root_mse, rec_root_mse, nonrec_root_mse, all_mean_ape, rec_mean_ape, nonrec_mean_ape, attn_weights, inc_accu, inc_precision, inc_recall
            


def infer_last_obs(infer_dataloader, device):
    '''
    FUNCTION 
        One of the Baseline Methods
        Make Inference Based on Latest Observation and Compute Loss

    INPUTs
        infer_dataloader: dataloader of inference data
        device: "cuda" or "cpu"
    
    OUTPUTs:
        root_mse: size (out_seq_len), root mean square error for each output time slot
        mean_ape: size (out_seq_len), mean absolute percentage error for each output time slot
    '''
    all_square_error, rec_square_error, nonrec_square_error = 0, 0, 0
    all_abs_perc_error, rec_abs_perc_error, nonrec_abs_perc_error = 0, 0, 0
    instance_cnt, rec_cnt, nonrec_cnt = 0, 0, 0

    with torch.no_grad():
        for i, batch in enumerate(infer_dataloader):
            _, target, recurrent = batch
            target = target.to(args.device)  # (batch_size, out_seq_len + 1, out_dim, 2) for TrafficModel, (batch_size, out_seq_len + 1, out_dim) for TrafficSeq2Seq
            recurrent = recurrent.to(args.device)  # (batch_size) 

            instance_cnt += target.size(0)
            rec_cnt += torch.sum(recurrent)
            nonrec_cnt += instance_cnt - rec_cnt

            # Make Prediction
            spd_pred = target[:, 0, :, 0].unsqueeze(1).repeat(1,args.out_seq_len,1)

            # Compute Intermediate Error
            all_square_error += torch.sum((spd_pred-target[:, 1:, :, 0])**2, axis=0)  # (out_seq_len, out_dim)
            rec_square_error += torch.sum(((spd_pred-target[:, 1:, :, 0])**2)[recurrent==1], axis=0)  # (out_seq_len, out_dim)
            nonrec_square_error += torch.sum(((spd_pred-target[:, 1:, :, 0])**2)[recurrent==0], axis=0)  # (out_seq_len, out_dim)

            all_abs_perc_error += torch.sum(torch.abs(spd_pred-target[:, 1:, :, 0])/target[:, 1:, :, 0], axis=0)  # (out_seq_len, out_dim)
            rec_abs_perc_error += torch.sum(torch.abs(spd_pred[recurrent==1]-target[:, 1:, :, 0][recurrent==1])/target[:, 1:, :, 0][recurrent==1], axis=0)  # (out_seq_len, out_dim)
            nonrec_abs_perc_error += torch.sum(torch.abs(spd_pred[recurrent==0]-target[:, 1:, :, 0][recurrent==0])/target[:, 1:, :, 0][recurrent==0], axis=0)  # (out_seq_len, out_dim)
        
        # Evaluate Speed Prediction (and Incident Prediction), and Computer Average Attention Weight
        all_root_mse = (torch.sum(all_square_error, axis=1)/(instance_cnt*args.out_dim))**0.5  # (out_seq_len)
        rec_root_mse = (torch.sum(rec_square_error, axis=1)/(rec_cnt*args.out_dim))**0.5  # (out_seq_len)
        nonrec_root_mse = (torch.sum(nonrec_square_error, axis=1)/(nonrec_cnt*args.out_dim))**0.5  # (out_seq_len)

        all_mean_ape = torch.sum(all_abs_perc_error, axis=1)/(instance_cnt*args.out_dim)  # (out_seq_len)
        rec_mean_ape = torch.sum(rec_abs_perc_error, axis=1)/(rec_cnt*args.out_dim)  # (out_seq_len)
        nonrec_mean_ape = torch.sum(nonrec_abs_perc_error, axis=1)/(nonrec_cnt*args.out_dim)  # (out_seq_len)

    return all_root_mse, rec_root_mse, nonrec_root_mse, all_mean_ape, rec_mean_ape, nonrec_mean_ape


def main(args):
    '''
    Load Model, Make Inference and Visualize Results if Applicable
    '''
    # 1. Create Directories
    create_dir(args.log_dir)
    create_dir(f"{args.log_dir}/{args.exp_name}")

    # 2. Set up Logger
    logging.basicConfig(filename=f"{args.log_dir}/{args.exp_name}/inference.log", filemode="w", format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG) 

    # 3. log experiment and data information
    logging.info('{:*^100}'.format(" COMMAND LINE "))
    logging.info(" ".join(sys.argv) + "\n")

    logging.info('{:*^100}'.format(" EXPERIMENT INFORMATION "))
    logging.info(f"Experiment Name: {args.exp_name}")
    logging.info(f"Batch Size: {args.batch_size} \n")

    logging.info('{:*^100}'.format(" DATA INFORMATION "))
    logging.info(f"Use Density: {args.use_density}; Use Truck Speed: {args.use_truck_spd}; Use Personal Vehicle Speed: {args.use_pv_spd}")
    logging.info(f"Input Sequence Length: {args.in_seq_len}; Output Sequence Lenth: {args.out_seq_len}; Output Frequency: {args.out_freq} \n")

    logging.info('{:*^100}'.format(" LOADING PROGRESS "))

    # 4. Initialize Models
    base_model = TrafficSeq2Seq(args)
    traffic_model = TrafficModel(args)
    
    # 5. Load Model Checkpoint
    base_model_path = "{}/base/best_{}.pt".format(args.checkpoint_dir, args.exp_name)
    traffic_model_path = "{}/finetune/best_{}.pt".format(args.checkpoint_dir, args.exp_name)
    with open(base_model_path, "rb") as f_base_model, open(traffic_model_path, "rb") as f_traffic_model:
        # load state dict
        state_dict_base = torch.load(f_base_model, map_location=args.device)
        state_dict_traffic = torch.load(f_traffic_model, map_location=args.device)

        # populate state dict into models and log
        base_model.load_state_dict(state_dict_base)
        logging.info(f"successfully loaded checkpoint from {base_model_path}")

        traffic_model.load_state_dict(state_dict_traffic)
        logging.info(f"successfully loaded checkpoint from {traffic_model_path}")
    
    # 6. Load Data for Inference
    infer_dataloader = get_inference_data_loader(args)
    logging.info(f"successfully loaded data \n")

    # 7. Get Inference Result
    logging.info('{:*^100}'.format(" EXPERIMENT RESULT "))
    base_all_root_mse, base_rec_root_mse, base_nonrec_root_mse, base_all_mean_ape, base_rec_mean_ape, base_nonrec_mean_ape, base_attn_weight = infer(infer_dataloader=infer_dataloader, model=base_model, model_type="base", args=args)
    traffic_all_root_mse, traffic_rec_root_mse, traffic_nonrec_root_mse, traffic_all_mean_ape, traffic_rec_mean_ape, traffic_nonrec_mean_ape, [traffic_LR_attn_weight, traffic_rec_attn_weight, traffic_nonrec_attn_weight], inc_accu, inc_precision, inc_recall = infer(infer_dataloader=infer_dataloader, model=traffic_model, model_type="traffic", args=args)
    lo_all_root_mse, lo_rec_root_mse, lo_nonrec_root_mse, lo_all_mean_ape, lo_rec_mean_ape, lo_nonrec_mean_ape = infer_last_obs(infer_dataloader=infer_dataloader, device=args.device)

    logging.info('{:=^100}'.format(" Traffic Model "))
    logging.info(f"RMSE - all: {traffic_all_root_mse},  recurrent: {traffic_rec_root_mse},  nonrecurrent: {traffic_nonrec_root_mse}")
    logging.info(f"MAPE - all: {traffic_all_mean_ape},  recurrent: {traffic_rec_mean_ape},  nonrecurrent: {traffic_nonrec_mean_ape}")
    logging.info(f"Incident Prediction - Accuracy:{inc_accu},  Precision:{inc_precision},  Recall:{inc_recall}")

    logging.info('{:=^100}'.format(" Baseline 1 - Seq2Seq "))
    logging.info(f"RMSE - all: {base_all_root_mse},  recurrent: {base_rec_root_mse},  nonrecurrent: {base_nonrec_root_mse}")
    logging.info(f"MAPE - all: {base_all_mean_ape},  recurrent: {base_rec_mean_ape},  nonrecurrent: {base_nonrec_mean_ape}")

    logging.info('{:=^100}'.format(" Baseline 2 - Lateset Observation "))
    logging.info(f"RMSE - all: {lo_all_root_mse},  recurrent: {lo_rec_root_mse},  nonrecurrent: {lo_nonrec_root_mse}")
    logging.info(f"MAPE - all: {lo_all_mean_ape},  recurrent: {lo_rec_mean_ape},  nonrecurrent: {lo_nonrec_mean_ape}")

    logging.info('{:=^100}'.format(" Baseline 1 - Historical Average "))
    logging.info("RMSE - 5.104168739538521")
    logging.info("MAPE - 0.12455292714700887")

    # 8. Visualize Attention Weights
    base_attn_weight_path = f"{args.log_dir}/{args.exp_name}/base_attn_weight.jpg"
    base_attn_weight_title = f"Attention Weight of Seq2Seq {args.exp_name}"

    traffic_LR_attn_weight_path = f"{args.log_dir}/{args.exp_name}/traffic_LR_attn_weight.jpg"
    traffic_LR_attn_weight_title = f"Attention Weight of LR Module in Traffic Model {args.exp_name}"
    traffic_rec_attn_weight_path = f"{args.log_dir}/{args.exp_name}/traffic_rec_attn_weight.jpg"
    traffic_rec_attn_weight_title = f"Attention Weight of Recurrent Decoder in Traffic Model {args.exp_name}"
    traffic_nonrec_attn_weight_path = f"{args.log_dir}/{args.exp_name}/traffic_nonrec_attn_weight.jpg"
    traffic_nonrec_attn_weight_title = f"Attention Weight of Nonrecurrent Decoder in Traffic Model {args.exp_name}"

    visualize_attn_weight(base_attn_weight, args, base_attn_weight_title, base_attn_weight_path)
    visualize_attn_weight(traffic_LR_attn_weight, args, traffic_LR_attn_weight_title, traffic_LR_attn_weight_path)
    visualize_attn_weight(traffic_rec_attn_weight, args, traffic_rec_attn_weight_title, traffic_rec_attn_weight_path)
    visualize_attn_weight(traffic_nonrec_attn_weight, args, traffic_nonrec_attn_weight_title, traffic_nonrec_attn_weight_path)

    traffic_spd_attn_weight = (traffic_rec_attn_weight + traffic_nonrec_attn_weight)/2
    traffic_spd_attn_weight_path = f"{args.log_dir}/{args.exp_name}/traffic_spd_attn_weight.jpg"
    traffic_spd_attn_weight_title = f"Attention Weight of Speed Prediction in Traffic Model {args.exp_name}"
    visualize_attn_weight(traffic_spd_attn_weight, args, traffic_spd_attn_weight_title, traffic_spd_attn_weight_path)

    logging.info(f"Please check visualizations of attention weights under folder {args.log_dir}/{args.exp_name}")
    

if __name__ == "__main__":

    # 1. Modify Arguments
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # For reproducibility
    torch.manual_seed(args.seed)

    # Task specific directories
    args.log_dir = "./results" 
    args.exp_name += f"_{str(args.use_density)[0]}_{str(args.use_truck_spd)[0]}_{str(args.use_pv_spd)[0]}_{args.in_seq_len}_{args.out_seq_len}_{args.out_freq}" 

    # Change input dimension based on task type and whether to use new features or not
    if not args.use_density:
        args.in_dim -= 233
    
    if not args.use_truck_spd:
        args.in_dim -= 233
    
    if not args.use_pv_spd:
        args.in_dim -= 233
    
    main(args)
