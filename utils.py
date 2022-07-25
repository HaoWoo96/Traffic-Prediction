import os
import torch
import logging
import sys

import matplotlib.pyplot as plt
import seaborn as sns

########################################
#               FILE I/O               #
########################################
def save_checkpoint(epoch, model, args, best=False):
    if best:
        path = os.path.join(args.checkpoint_dir, 'best_{}.pt'.format(args.exp_name))
    else:
        path = os.path.join(args.checkpoint_dir, 'epoch_{}_{}.pt'.format(epoch, args.exp_name))
    torch.save(model.state_dict(), path)


def create_dir(directory):
    """
    Creates a directory if it does not already exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


########################################
#              VISUALIZATION           #
########################################
def visualize_attn_weight(attn_weight, args, title, save_path):
    '''
    FUNCTION
        Visualize Attention Weight Matrix
    '''
    # visualize attention weight matrix as a heat map
    img = sns.heatmap(attn_weight.T.flip(dims=(0,)),linewidths=.5, vmax=1)  # transpose and flip tensor to match it with axis direction of the plot

    # x-axis
    img.xaxis.set_ticks_position("top")
    img.xaxis.set_tick_params(direction='out', pad=5)
    img.set_xticklabels([f"t + {i*args.out_freq} min" for i in range(1,args.out_seq_len+1)], ha="left", rotation=45, rotation_mode='anchor')

    # x-label
    img.xaxis.set_label_position('top')
    img.set_xlabel("Output Sequence", loc="right")

    # y-axis
    img.yaxis.set_ticks_position("left")
    img.yaxis.set_tick_params(direction='out', pad=10)
    img.set_yticklabels([f"t - {i*5} min" if i>0 else "t" for i in range(args.in_seq_len)], rotation=360)

    # y-label
    img.yaxis.set_label_position('left')
    img.set_ylabel("Input Sequence", loc="bottom")
    img.set_title(title, y=0, pad=-25)

    # save figure
    img.get_figure().savefig(save_path, bbox_inches="tight")
    plt.figure()


########################################
#                LOGGING               #
########################################
def log_train_meta(args):
    '''
    FUNCTION
        Log meta info of training experiment
    '''
    # log experiment and data information
    logging.info('{:*^100}'.format(" COMMAND LINE "))
    logging.info(" ".join(sys.argv) + "\n")

    logging.info('{:*^100}'.format(" EXPERIMENT INFORMATION "))
    logging.info(f"Task: {args.task}")
    logging.info(f"Experiment Name: {args.exp_name}")
    logging.info(f"Number of Epochs: {args.num_epochs}, Learning Rate: {args.lr}, Batch Size: {args.batch_size} \n")
    # logging.info(f"Learning Rate: {args.lr}, Scheduler Decay Rate: {args.lr_decay_rate}, Scheduler Decay Frequency: {args.lr_decay_freq} \n")

    logging.info('{:*^100}'.format(" MODEL INFORMATION "))
    logging.info('(only used in finetune task):')
    logging.info(f"     Use Expectation of Prediction as Output: {args.use_expectation} ")
    logging.info(f"     Incident Threshold: {args.inc_threshold} (only used in finetune task)")
    # logging.info(f"Dropout Probability: {args.dropout_prob}")
    logging.info(f"Teacher Forcing Ratio: {args.teacher_forcing_ratio} \n")

    logging.info('{:*^100}'.format(" DATA INFORMATION "))
    logging.info(f"Ground Truth Speed Data Source: {args.gt_type}")
    logging.info(f"Use Density: {args.use_density}; Use Truck Speed: {args.use_truck_spd}; Use Personal Vehicle Speed: {args.use_pv_spd}; Use Raw Speed: {args.use_speed}")
    logging.info(f"Input Sequence Length: {args.in_seq_len}; Output Sequence Lenth: {args.out_seq_len}; Output Frequency: {args.out_freq} \n")

    logging.info('{:*^100}'.format(" LOADING PROGRESS "))


def log_eval_meta(args):
    '''
    FUNCTION
        Log meta info of model evaluation
    '''
    # log experiment and data information
    logging.info('{:*^100}'.format(" COMMAND LINE "))
    logging.info(" ".join(sys.argv) + "\n")

    logging.info('{:*^100}'.format(" TRAINING INFORMATION "))
    logging.info(f"Experiment Name: {args.exp_name}")
    logging.info(f"Learning Rate: {args.lr}, Batch Size: {args.batch_size} \n")

    logging.info('{:*^100}'.format(" 2-STAGE MODEL INFORMATION "))
    logging.info('(only used in finetune task):')
    logging.info(f"     Use Expectation of Prediction as Output: {args.use_expectation} ")
    logging.info(f"     Incident Threshold: {args.inc_threshold} (only used in finetune task)")
    # logging.info(f"Dropout Probability: {args.dropout_prob}")
    logging.info(f"Teacher Forcing Ratio: {args.teacher_forcing_ratio} \n")

    logging.info('{:*^100}'.format(" DATA INFORMATION "))
    logging.info(f"Ground Truth Speed Data Source: {args.gt_type}")
    logging.info(f"Use Density: {args.use_density}; Use Truck Speed: {args.use_truck_spd}; Use Personal Vehicle Speed: {args.use_pv_spd}; Use Raw Speed: {args.use_speed}")
    logging.info(f"Input Sequence Length: {args.in_seq_len}; Output Sequence Lenth: {args.out_seq_len}; Output Frequency: {args.out_freq} \n")

    logging.info('{:*^100}'.format(" LOADING PROGRESS "))


def log_eval_spd_result_xd(all_root_mse, rec_root_mse, nonrec_root_mse, all_mean_ape, rec_mean_ape, nonrec_mean_ape):
    '''
    FUNCTION
        Log evaluation result of speed prediction (ground truth: XD)
    '''
    logging.info(f"RMSE - all: {all_root_mse},  recurrent: {rec_root_mse},  nonrecurrent: {nonrec_root_mse}")
    logging.info(f"MAPE - all: {all_mean_ape},  recurrent: {rec_mean_ape},  nonrecurrent: {nonrec_mean_ape}")


def log_eval_spd_result_tmc(all_root_mse, rec_root_mse, nonrec_root_mse, all_mean_ape, rec_mean_ape, nonrec_mean_ape):
    '''
    FUNCTION
        Log evaluation result of speed prediction (ground truth: TMC)
    '''
    logging.info(f"[RMSE]")
    logging.info(' {}|{: ^60}|{: ^60}|{: ^60}|'.format(" "*20, "All Cases","Recurrent Cases", "Nonrecurrent Cases"))
    logging.info('|{: ^20}|{: ^60}|{: ^60}|{: ^60}|'.format("Overall", \
        str(torch.mean(torch.stack(list(all_root_mse.values())), axis=0)),\
        str(torch.mean(torch.stack(list(rec_root_mse.values())), axis=0)),\
        str(torch.mean(torch.stack(list(nonrec_root_mse.values())), axis=0))))
    logging.info('|{: ^20}|{: ^60}|{: ^60}|{: ^60}|'.format("All Vehicles", \
        str(all_root_mse["all"]),\
        str(rec_root_mse["all"]),\
        str(nonrec_root_mse["all"])))
    logging.info('|{: ^20}|{: ^60}|{: ^60}|{: ^60}|'.format("Trucks", \
        str(all_root_mse["truck"]),\
        str(rec_root_mse["truck"]),\
        str(nonrec_root_mse["truck"])))
    logging.info('|{: ^20}|{: ^60}|{: ^60}|{: ^60}|'.format("Personal Vehicles", \
        str(all_root_mse["pv"]),\
        str(rec_root_mse["pv"]),\
        str(nonrec_root_mse["pv"])))

    logging.info(" ")
    
    logging.info(f"[MAPE]")
    logging.info(' {}|{: ^60}|{: ^60}|{: ^60}|'.format(" "*20, "All Cases","Recurrent Cases", "Nonrecurrent Cases"))
    logging.info('|{: ^20}|{: ^60}|{: ^60}|{: ^60}|'.format("Overall", \
        str(torch.mean(torch.stack(list(all_mean_ape.values())), axis=0)),\
        str(torch.mean(torch.stack(list(rec_mean_ape.values())), axis=0)),\
        str(torch.mean(torch.stack(list(nonrec_mean_ape.values())), axis=0))))
    logging.info('|{: ^20}|{: ^60}|{: ^60}|{: ^60}|'.format("All Vehicles", \
        str(all_mean_ape["all"]),\
        str(rec_mean_ape["all"]),\
        str(nonrec_mean_ape["all"])))
    logging.info('|{: ^20}|{: ^60}|{: ^60}|{: ^60}|'.format("Trucks", \
        str(all_mean_ape["truck"]),\
        str(rec_mean_ape["truck"]),\
        str(nonrec_mean_ape["truck"])))
    logging.info('|{: ^20}|{: ^60}|{: ^60}|{: ^60}|'.format("Personal Vehicles", \
        str(all_mean_ape["pv"]),\
        str(rec_mean_ape["pv"]),\
        str(nonrec_mean_ape["pv"])))

    logging.info(" ")
