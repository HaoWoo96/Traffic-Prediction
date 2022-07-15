import os
import torch

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