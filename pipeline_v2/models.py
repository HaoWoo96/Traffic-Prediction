import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer

from modules import EncoderRNN, EncoderTrans, PosEmbed, DecoderRNN, DecoderMLP, AttnDecoderRNN, STBlock

# 1. Sequence-to-sequence Model without Factorization
class Seq2SeqNoFact(nn.Module):
    '''
    Proposed seq2seq model in "Learning to Recommend Signal Plans under Incidents with Real-Time Traffic Prediction"
    We use it here as a baseline method
    '''
    def __init__(self, args):
        super(Seq2SeqNoFact, self).__init__()
        self.args = args
        self.encoder = EncoderRNN(args=args)
        self.decoder = AttnDecoderRNN(args=args, dec_type="Non_LR")
    
    def forward(self, x, target):
        '''
        INPUTs
            x: input, (batch_size, seq_len_in, dim_in)
            target: (batch_size, seq_len_out, dim_out), for Seq2SeqNoFact, we pass in the speed data of <all vehicles>, which is index 0 in the last dimension of Y

        OUTPUTs
            dec_out: (batch_size, seq_len_out, dim_out)  
            dec_hidden: (num_layer, batch_size, dim_hidden)
            attn_weights: (batch_size, seq_len_out, seq_len_in)
        '''
        batch_size = x.size(0)

        # pass into encoder
        ini_hidden = self.encoder.initHidden(batch_size = batch_size)
        enc_out, enc_hidden = self.encoder(x, ini_hidden)

        # pass into decoder
        dec_out, dec_hidden, attn_weights = self.decoder(target[..., 0], enc_hidden ,enc_out)

        return dec_out, dec_hidden, attn_weights

# 2. Sequence-to-sequence Model with Factorization
class Seq2SeqFact(nn.Module):
    def __init__(self, args):
        super(Seq2SeqFact, self).__init__()
        self.args = args
        self.encoder = EncoderRNN(args=args)

        self.LR_decoder = AttnDecoderRNN(args=args, dec_type="LR")  # decoder for incident prediction (logistic regression)
        self.rec_decoder = AttnDecoderRNN(args=args, dec_type="Non_LR")  # decoder for speed prediction in recurrent scenario
        self.nonrec_decoder = AttnDecoderRNN(args=args, dec_type="Non_LR")  # decoder for speed prediction in non-recurrent scenario
    
    def forward(self, x, target):
        '''
        INPUTs
            x: input, (batch_size, seq_len_in, num_feat_dim + incident_feat_dim)
            target: (batch_size, seq_len_out + 1, dim_out, 4), the last dimension refers to 1~3: speed (all, truck, pv) and 4: incident status 

        OUTPUTs
            LR_out: incident status prediction, (batch_size, seq_len_out, dim_out)
            speed_pred: speed prediction, (batch_size, seq_len_out, dim_out or 3*dim_out)
            xxx_attn_weights: (batch_size, seq_len_out, seq_len_in)
        '''
        batch_size = x.size(0)

        # pass into encoder
        ini_hidden = self.encoder.initHidden(batch_size=batch_size)
        enc_out, enc_hidden = self.encoder(x, ini_hidden)

        # pass into decoder
        '''
        xxx_dec_out: (batch_size, seq_len_out, dim_out)  
        xxx_dec_hidden: (num_layer, batch_size, dim_hidden)
        xxx_attn_weights: (batch_size, seq_len_out, seq_len_in)
        '''
        # print(target[..., 1].size(), enc_hidden.size(), enc_out.size())
        LR_out, LR_hidden, LR_attn_weights = self.LR_decoder(target[..., 3], enc_hidden, enc_out)
        rec_out, rec_hidden, rec_attn_weights = self.rec_decoder(target[..., 0], enc_hidden, enc_out)
        nonrec_out, nonrec_hidden, nonrec_attn_weights = self.nonrec_decoder(target[..., 0], enc_hidden, enc_out)

        # generate final speed prediction
        if self.args.use_gt_inc:
                speed_pred = rec_out * (target[..., 3] == 0.0) + nonrec_out * (target[..., 3] == 1.0)
        else:
            if self.args.use_expectation:
                rec_weight = torch.ones(LR_out.size()).to(self.args.device) - LR_out
                speed_pred = rec_out * rec_weight + nonrec_out * LR_out 
            else:
                speed_pred = rec_out * (LR_out <= self.args.inc_threshold) + nonrec_out * (LR_out > self.args.inc_threshold)


        if self.args.task == "LR":
            return LR_out, LR_hidden, LR_attn_weights
        elif self.args.task == "rec":
            return rec_out, rec_hidden, rec_attn_weights
        elif self.args.task == "nonrec":
            return nonrec_out, nonrec_hidden, nonrec_attn_weights
        else:
            return speed_pred, LR_out, [LR_attn_weights, rec_attn_weights, nonrec_attn_weights]

# 3. Transformer Model without Factorization
class TransNoFact(nn.Module):
    def __init__(self, args):
        super(TransNoFact, self).__init__()
        self.encoder = EncoderTrans(args)
        self.decoder = DecoderMLP(args, "R")
        self.args = args
    
    def forward(self, x, target):
        '''
        INPUTs
            x: input, (batch_size, seq_len_in, dim_in)

        OUTPUTs
            dec_out: (batch_size, seq_len_out, dim_out or 3*dim_out)  
        '''
        enc_out = self.encoder(x)  # (batch_size, seq_len_in, dim_in)
        dec_out = self.decoder(enc_out)

        return dec_out


# 4. Transformer Model with Factorization
class TransFact(nn.Module):
    def __init__(self, args):
        super(Seq2SeqFact, self).__init__()
        self.args = args
        self.encoder = EncoderTrans(args=args)

        self.LR_decoder = DecoderMLP(args=args, dec_type="LR")  # decoder for incident prediction (logistic regression)
        self.rec_decoder = DecoderMLP(args=args, dec_type="Non_LR")  # decoder for speed prediction in recurrent scenario
        self.nonrec_decoder = DecoderMLP(args=args, dec_type="Non_LR")  # decoder for speed prediction in non-recurrent scenario
    
    def forward(self, x, target):
        '''
        INPUTs
            x: input, (batch_size, seq_len_in, dim_in)
            target: (batch_size, seq_len_out + 1, dim_out, 2), the last dimension refers to 1. speed and 2. incident status 

        OUTPUTs
            LR_out: incident status prediction, (batch_size, seq_len_out, dim_out)
            speed_pred: speed prediction, (batch_size, seq_len_out, dim_out or 3*dim_out)
        '''

        # pass into encoder
        enc_out = self.encoder(x)  # (batch_size, seq_len_in, dim_in)

        # pass into decoder
        '''
        xxx_dec_out: (batch_size, seq_len_out, dim_out)  
        '''
        LR_out = self.LR_decoder(enc_out)
        rec_out = self.rec_decoder(enc_out)
        nonrec_out = self.nonrec_decoder(enc_out)

        if self.args.gt_type == "tmc":
            # generate final speed prediction
            if self.args.use_gt_inc:  
                # use ground truth incident prediction as 0-1 mask
                speed_pred = rec_out * (target[..., 3].repeat(1,1,3) <= 0.5) + nonrec_out * (target[..., 3].repeat(1,1,3) > 0.5)
            else:
                if self.args.use_expectation:
                    # compute speed prediction as expectation
                    rec_weight = torch.ones(LR_out.repeat(1,1,3).size()).to(self.args.device) - LR_out.repeat(1,1,3)
                    speed_pred = rec_out * rec_weight + nonrec_out * LR_out.repeat(1,1,3) 
                else:
                    # compute speed prediction based on predicted 0-1 mask
                    speed_pred = rec_out * (LR_out.repeat(1,1,3) <= self.args.inc_threshold) + nonrec_out * (LR_out.repeat(1,1,3) > self.args.inc_threshold)
       
        else:
            # generate final speed prediction
            if self.args.use_gt_inc:
                    speed_pred = rec_out * (target[..., 3] < 1.0) + nonrec_out * (target[..., 3] >= 1.0)
            else:
                if self.args.use_expectation:
                    rec_weight = torch.ones(LR_out.size()).to(self.args.device) - LR_out
                    speed_pred = rec_out * rec_weight + nonrec_out * LR_out 
                else:
                    speed_pred = rec_out * (LR_out <= self.args.inc_threshold) + nonrec_out * (LR_out > self.args.inc_threshold)


        if self.args.task == "LR":
            return LR_out
        elif self.args.task == "rec":
            return rec_out
        elif self.args.task == "nonrec":
            return nonrec_out
        else:
            return speed_pred, LR_out



# 5. Spatial-temporal Model with GCN+Transformer Encoder
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.learnable_embed = nn.init.normal_(torch.empty((args.dim_in - args.in_fixed_dim, ), requires_grad=True, device=args.device))
        self.in_linear = nn.Linear(args.dim_in, args.dim_hidden)
        self.spatial_temporal_blocks = nn.Sequential(
            *[STBlock(args) for _ in range(args.num_block)]
        )
        self.conv = nn.Conv2d(in_channels=args.seq_len_in, out_channels=args.seq_len_out, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.out_linear = nn.Linear(args.dim_hidden+6, args.dim_out)

    def forward(self, x):
        '''
        INPUT
            x: tuple of (weekly_avg_node_embedding, daily_avg_node_embedding, current_node_embedding)
                weekly_avg_node_embedding: node embedding of, size (batch_size, seq_len_out, num_node, 3) 
                daily_avg_node_embedding: node embedding of, size (batch_size, seq_len_out, num_node, 3)
                current_node_embedding: initial node embedding of current tracking period, size (batch_size, seq_len_in, num_node, in_fixed_dim)
        
        OUTPUT
            result: size (batch_size, seq_len_out, num_node, dim_out)
        '''
        weekly_avg_node_embedding, daily_avg_node_embedding, current_node_embedding = x
        batch_size, seq_len_in, num_node, _ = current_node_embedding.shape

        # 1. Concat Learnable Embedding
        learnable_embed = self.learnable_embed.repeat((batch_size, seq_len_in, num_node, 1))  # (batch_size, seq_len_in, num_node, dim_in - in_fixed_dim)
        new_current_node_embedding = torch.cat((current_node_embedding, learnable_embed), dim=3) # (batch_size, seq_len_in, num_node, dim_in)

        # 2. Spatial-temporal Feature Extraction
        result = self.in_linear(new_current_node_embedding) # (batch_size, seq_len_in, num_node, dim_hidden)
        result = self.spatial_temporal_blocks(result) # (batch_size, seq_len_in, num_node, dim_hidden)

        # 3. Convolution along the Axis of Time
        result = self.relu(self.conv(result)) # (batch_size, seq_len_out, num_node, dim_hidden)
        result = torch.cat((weekly_avg_node_embedding, daily_avg_node_embedding, result), dim=3)  # (batch_size, seq_len_out, num_node, dim_hidden+6)
        result = self.out_linear(result) # (batch_size, seq_len_out, num_node, dim_out)
        
        return result


