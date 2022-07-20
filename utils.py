import os
import torch

import matplotlib.pyplot as plt
import seaborn as sns

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