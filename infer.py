import numpy as np
import argparse
import torch
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from models import TrafficSeq2Seq, TrafficModel
from data_loader import get_data_loader
from utils import save_checkpoint, create_dir
from train import create_parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    args.checkpoint_dir = args.checkpoint_dir+"/"+args.task # checkpoint directory is task specific

    # Initialize Model


    # Load Model Checkpoint


    # Make Prediction 


    # Compute Loss & Various Metrics


    # Visualize Attention Weights
