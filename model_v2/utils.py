import os
import torch
import logging
import sys
import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

########################################
#            Model Training            #
########################################
# for reproducibility
def seed_torch(seed=912):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


########################################
#            Model Structure           #
########################################
class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out[1])

    def clear(self):
        self.outputs = []

def patch_attention(m):
    """
    Reference: https://gist.github.com/airalcorn2/50ec06517ce96ecc143503e21fa6cb91

    Helper function to allow transformer to output attention matrix for viz.
    Current PyTorch implementaiton of nn.Transformer doesn't output attention matrix, as it sets "need_weights" option of self/multihead attention modules = False
    See for more details: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoder.forward

    Example of calling patch_attention:
        save_output = SaveOutput()
        for module in transformer.modules():
            if isinstance(module, nn.MultiheadAttention):
                utils.patch_attention(module)
                module.register_forward_hook(save_output)
        with torch.no_grad():
            pred = transformer(x, y)
        
        # num of multihead attn layers * batch_size
        # to get the attn weight matrices between in_seq and out_seq
        # index the multihead attn layer(s) and then index the batch
        # You can set "average_attn_weights" = True if needed
        print(save_output) 
    """
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs['need_weights'] = True
        kwargs['average_attn_weights'] = False

        return forward_orig(*args, **kwargs)

    m.forward = wrap

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
    img.set_xticklabels([f"t + {i*args.freq_out} min" for i in range(1,args.seq_len_out+1)], ha="left", rotation=45, rotation_mode='anchor')

    # x-label
    img.xaxis.set_label_position('top')
    img.set_xlabel("Output Sequence", loc="right")

    # y-axis
    img.yaxis.set_ticks_position("left")
    img.yaxis.set_tick_params(direction='out', pad=10)
    img.set_yticklabels([f"t - {i*5} min" if i>0 else "t" for i in range(args.seq_len_in)], rotation=360)

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
    logging.info(f"     Incident Threshold: {args.inc_threshold}")
    # logging.info(f"Dropout Probability: {args.dropout_prob}")
    logging.info(f"Teacher Forcing Ratio: {args.teacher_forcing_ratio} \n")

    logging.info('{:*^100}'.format(" DATA INFORMATION "))
    #logging.info(f"Ground Truth Speed Data Source: {args.gt_type}")
    logging.info(f"Use Density: {args.use_dens}; Use All Speed: {args.use_spd_all}; Use Truck Speed: {args.use_spd_truck}; Use Personal Vehicle Speed: {args.use_spd_pv}; ")
    logging.info(f"Input Sequence Length: {args.seq_len_in}; Output Sequence Lenth: {args.seq_len_out}; Output Frequency: {args.freq_out} \n")

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
    logging.info(f"     Incident Threshold: {args.inc_threshold} ")
    # logging.info(f"Dropout Probability: {args.dropout_prob}")
    logging.info(f"Teacher Forcing Ratio: {args.teacher_forcing_ratio} \n")

    logging.info('{:*^100}'.format(" DATA INFORMATION "))
    logging.info(f"Use Density: {args.use_dens}; Use Raw Speed: {args.use_spd_all}; Use Truck Speed: {args.use_spd_truck}; Use Personal Vehicle Speed: {args.use_spd_pv}")
    logging.info(f"Input Sequence Length: {args.seq_len_in}; Output Sequence Lenth: {args.seq_len_out}; Output Frequency: {args.freq_out} \n")

    logging.info('{:*^100}'.format(" LOADING PROGRESS "))


def log_eval_spd_result_xd(all_root_mse, rec_root_mse, nonrec_root_mse, all_mean_ape, rec_mean_ape, nonrec_mean_ape):
    '''
    FUNCTION
        Log evaluation result of speed prediction (ground truth: XD)
    '''
    logging.info(f"RMSE - all: {all_root_mse},  recurrent: {rec_root_mse},  nonrecurrent: {nonrec_root_mse}")
    logging.info(f"MAPE - all: {all_mean_ape},  recurrent: {rec_mean_ape},  nonrecurrent: {nonrec_mean_ape}")

def log_lasso_result_xd(all_root_mse, rec_root_mse, nonrec_root_mse, all_mean_ape, rec_mean_ape, nonrec_mean_ape):
    '''
    FUNCTION
        Log LASSO result of speed prediction (ground truth: XD)
    '''
    logging.info(f"RMSE - all: {[6.18508285, 6.18535092, 6.18686026, 6.18816916, 6.18909456, 6.19130026]},  recurrent: {[]},  nonrecurrent: {[]}")
    logging.info(f"MAPE - all: {[0.14234066, 0.14231179, 0.14230198, 0.14237242, 0.14240318, 0.14245276]},  recurrent: {[]},  nonrecurrent: {[]}")

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


def log_lasso_result_tmc():
    '''
    FUNCTION
        Log LASSO regression result of speed prediction (ground truth: TMC)
    '''
    logging.info(f"[RMSE]")
    logging.info(' {}|{: ^60}|{: ^60}|{: ^60}|'.format(" "*20, "All Cases","Recurrent Cases", "Nonrecurrent Cases"))
    logging.info('|{: ^20}|{: ^60}|{: ^60}|{: ^60}|'.format("Overall", \
        None,\
        None,\
        ))
    logging.info('|{: ^20}|{: ^60}|{: ^60}|{: ^60}|'.format("All Vehicles", \
        [7.54326613, 7.74410466, 7.7462895 , 7.74468577, 7.74443766,
        7.7648749],\
        [],\
        []))
    logging.info('|{: ^20}|{: ^60}|{: ^60}|{: ^60}|'.format("Trucks", \
        [7.34729147, 7.4914095 , 7.52732259, 7.51603671, 7.52214061,
        7.52891235],\
        [],\
        []))
    logging.info('|{: ^20}|{: ^60}|{: ^60}|{: ^60}|'.format("Personal Vehicles", \
        [7.87156427, 8.01936315, 8.0423071 , 8.07119211, 8.04648996,
        8.04690939],\
        [],\
        []))

    logging.info(" ")
    
    logging.info(f"[MAPE]")
    logging.info(' {}|{: ^60}|{: ^60}|{: ^60}|'.format(" "*20, "All Cases","Recurrent Cases", "Nonrecurrent Cases"))
    logging.info('|{: ^20}|{: ^60}|{: ^60}|{: ^60}|'.format("Overall", \
        [],\
        [],\
        []))
    logging.info('|{: ^20}|{: ^60}|{: ^60}|{: ^60}|'.format("All Vehicles", \
        [0.19925693, 0.20524506, 0.20589553, 0.20583538, 0.20577297,
        0.20619039],\
        [],\
        []))
    logging.info('|{: ^20}|{: ^60}|{: ^60}|{: ^60}|'.format("Trucks", \
        [0.20684156, 0.21200868, 0.21337647, 0.21310218, 0.21324997,
        0.21346233],\
        [],\
        []))
    logging.info('|{: ^20}|{: ^60}|{: ^60}|{: ^60}|'.format("Personal Vehicles", \
        [0.2068008 , 0.21196173, 0.21317841, 0.2138334 , 0.21310976,
        0.2130874],\
        [],\
        []))

    logging.info(" ")
