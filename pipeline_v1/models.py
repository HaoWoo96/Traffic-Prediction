import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer


#############################
#       BUILDING BLOCKS     #
#############################
class EncoderRNN(nn.Module):
    def __init__(self, args):
        super(EncoderRNN, self).__init__()
        self.args = args

        # self.incident_start, self.incident_end = args.incident_indices  # starting and ending indices of categorical features (incident)
        # self.incident_embedding = nn.Embedding(num_embeddings=args.incident_range, embedding_dim=args.incident_embed_dim)
        self.input_processing = nn.Sequential(
                nn.Dropout(args.dropout_prob),
                nn.Linear(args.in_dim, 2*args.hidden_dim)
                )
        # self.gru = nn.GRU(input_size=2*args.hidden_dim, hidden_size=args.hidden_dim, num_layers=args.num_layer, batch_first=True)
        self.gru = nn.GRU(input_size=args.in_dim, hidden_size=args.hidden_dim, num_layers=args.num_layer, batch_first=True)

    def forward(self, x, hidden):
        '''
        INPUTs
            x: input, (batch_size, in_seq_len, in_dim)
            hidden: (num_layer, batch_size, hidden_dim)
        OUTPUTs
            output: (batch_size, in_seq_len, hidden_dim)
            hidden: (num_layer, batch_size, hidden_dim)
        '''
        # embedded_incident_feat = self.incident_embedding(input[:, :, self.incident_start:self.incident_end])  # (batch_size, in_seq_len, incident_feat_dim, incident_embed_dim)
        # embedded_incident_feat = torch.flatten(embedded_incident_feat, start_dim=-2)  # (batch_size, in_seq_len, incident_feat_dim * incident_embed_dim)
        # embedded_input = torch.cat((input[:self.incident_start], embedded_incident_feat, input[self.incident_end:]), dim=-1)  # (batch_size, in_seq_len, in_feat_dim)
        # output, hidden = self.gru(embedded_input, hidden)

        # processed_input = self.input_processing(x)
        # output, hidden = self.gru(processed_input, hidden)
        output, hidden = self.gru(x, hidden)

        return output, hidden

    def initHidden(self, batch_size):
        # here we supply an argument batch_size instead of using self.args.batch_size
        # because the last batch may not have full batch_size
        return torch.zeros(self.args.num_layer, batch_size, self.args.hidden_dim, device=self.args.device)


# Positional Embedding Encoder for Transformer Encoder
class PosEmbed(nn.Module):
    def __init__(self, args):
        super(PosEmbed, self).__init__()
        self.dropout = nn.Dropout(p=args.dropout)
        
        position = torch.arange(args.in_seq_len).unsqueeze(0)
        div_term = torch.exp(torch.arange(0, args.in_dim, 2)*(-math.log(10000.0) / args.in_dim))
        self.pe = torch.zeros(1, args.in_seq_len, args.in_dim)
        self.pe[0, :, 0::2] = torch.sin(position * div_term)
        self.pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', self.pe)
        self.args = args


    def forward(self, x):
        '''
        INPUTs
            x: input, (batch_size, in_seq_len, in_dim)
        OUTPUTs
            output: (batch_size, in_seq_len, in_dim)
        '''
        x = x + self.pe
        return self.dropout(x)


# Transformer Encoder
class EncoderTrans(nn.Module):
    def __init__(self, args):
        super(EncoderTrans, self).__init__()
        self.args = args

        self.pos_encoder = PosEmbed(args)
        self.trans_encoder_layers = TransformerEncoderLayer(d_model=args.in_dim, nhead=args.num_head, dropout=args.dropout, batch_first=True, norm_first=True)  # layer normalization should be first, otherwise the training will be very difficult
        self.trans_encoder = TransformerEncoder(encoder_layers=self.trans_encoder_layers, nlayers=args.num_trans_layers)

        # Generates an upper-triangular matrix of -inf, with zeros on diag.
        self.mask = torch.triu(torch.ones(args.in_seq_len, args.in_seq_len) * float('-inf'), diagonal=1)  # size (in_seq_len, in_seq_len)

    def forward(self, x):
        '''
        INPUTs
            x: input, (batch_size, in_seq_len, in_dim)
        OUTPUTs
            output: (batch_size, in_seq_len, in_dim)
        '''
        new_x = x * math.sqrt(self.args.in_dim)
        new_x = self.pos_encoder(new_x)
        output = self.transformer_encoder(new_x, self.mask)
        return output
        

# RNN Decoder without Attention
class DecoderRNN(nn.Module):
    def __init__(self, args, dec_type):
        '''
        INPUTs
            args: arguments
            dec_type: type of decoder ("LR": for logistic regression of incident occurrence; "R": for speed prediction)
        '''
        super(DecoderRNN, self).__init__()
        self.args = args
        self.dec_type = dec_type

        self.gru = nn.GRU(input_size=args.out_dim, hidden_size=args.hidden_dim, num_layers=args.num_layer, batch_first=True)
        
        self.out = nn.Sequential(
                nn.Linear(args.hidden_dim, args.out_dim),
                nn.Linear(args.out_dim, args.out_dim)
                )

    def forward(self, target, hidden):
        '''
        INPUTs
            target: (batch_size, out_seq_len, out_dim), the entries in the last dimension is either speed data or incident status
            hidden: the hidden tensor computed from encoder, (num_layer, batch_size, hidden_dim) 

        OUTPUTs
            output: (batch_size, out_seq_len, out_dim) 
            hidden: (num_layer, batch_size, hidden_dim)
        '''
        use_teacher_forcing = True if random.random() < self.args.teacher_forcing_ratio else False

        output = []
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for i in range(self.args.out_seq_len-1):
                x = target[:, i, :].unsqueeze(1)  # Teacher forcing, (batch_size, 1, out_dim)
                temp_out, hidden = self.gru(x, hidden)  # seq len is 1 for each gru operation
                output.append(self.out(temp_out))
        else:
            # Without teacher forcing: use its own predictions as the next input
            x = target[:, 0, :].unsqueeze(1)
            for i in range(self.args.out_seq_len-1):
                temp_out, hidden = self.gru(x, hidden)  # seq len is 1 for each gru operation
                temp_out = self.out(temp_out) 
                output.append(temp_out)

                x = temp_out.detach()  # use prediction as next input, detach from history
        
        return torch.cat(tensors=output, dim=1), hidden

# RNN Decoder with Attention
class AttnDecoderRNN(nn.Module):
    def __init__(self, args, dec_type):
        '''
        INPUTs
            args: arguments
            dec_type: type of decoder ("LR": for logistic regression of incident occurrence; "Non_LR": for speed prediction)
        '''
        super(AttnDecoderRNN, self).__init__()
        self.args = args

        # self.dropout = nn.Dropout(self.dropout_p)
        out_dim = args.out_dim

        # for TMC speed prediction, model should output 3*out_dim results (all/truck/personal vehicle)
        if args.gt_type == "tmc" and dec_type != "LR":
            out_dim = 3*args.out_dim

        self.attn_weight = nn.Linear(out_dim + args.hidden_dim, args.in_seq_len)
        self.attn_combine = nn.Linear(out_dim + args.hidden_dim, out_dim)

        self.gru = nn.GRU(input_size=out_dim, hidden_size=args.hidden_dim, num_layers=args.num_layer, batch_first=True)
        
        self.out = nn.Sequential(
                nn.Linear(args.hidden_dim, out_dim),
                nn.Linear(out_dim, out_dim)
                )
    

    def forward(self, target, hidden, enc_output):
        '''
        INPUTs
            target: (batch_size, out_seq_len, out_dim), the entries in the last dimension is either speed data or incident status
            hidden: the hidden tensor computed from encoder, (dec_num_layer, batch_size, hidden_dim) 
            enc_output: (batch_size, in_seq_len, hidden_dim)

        OUTPUTs
            output: (batch_size, out_seq_len, out_dim) 
            hidden: (num_layer, batch_size, hidden_dim)
            attn_weights: (batch_size, out_seq_len, in_seq_len)
        '''
        use_teacher_forcing = True if random.random() < self.args.teacher_forcing_ratio else False

        output = []
        attn_weights = []

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for i in range(self.args.out_seq_len):
                x = target[:, i, :].unsqueeze(1)  # Teacher forcing, (batch_size, 1, out_dim)

                # extract the top most hidden tensor, concat it with target input, and compute attention weights
                attn_weight = F.softmax(self.attn_weight(torch.cat(tensors=(x, hidden[-1, :, :].unsqueeze(1)), dim=2)), dim=2)  # (batch_size, 1, in_seq_len)
                attn_weights.append(attn_weight)
                
                # apply attention weights to encoder output
                weighted_enc_output = torch.bmm(attn_weight, enc_output)  # (batch_size, 1, hidden_dim)
                
                # concat weighted encoder output with target input 
                x = self.attn_combine(torch.cat(tensors=(x, weighted_enc_output), dim=2))  # (batch_size, 1, out_dim)
                x = F.relu(x)

                temp_out, hidden = self.gru(x, hidden)  # seq len is 1 for each gru operation
                output.append(self.out(temp_out))
        else:
            # Without teacher forcing: use its own predictions as the next input
            x = target[:, 0, :].unsqueeze(1)  # (batch_size, 1, out_dim)
            for i in range(self.args.out_seq_len):

                # extract the top most hidden tensor, concat it with target input, and compute attention weights
                attn_weight = F.softmax(self.attn_weight(torch.cat(tensors=(x, hidden[-1, :, :].unsqueeze(1)), dim=2)), dim=2)  # (batch_size, 1, in_seq_len)
                attn_weights.append(attn_weight)
                
                # apply attention weights to encoder output
                weighted_enc_output = torch.bmm(attn_weight, enc_output)  # (batch_size, 1, hidden_dim)
                
                # concat weighted encoder output with target input 
                x = self.attn_combine(torch.cat(tensors=(x, weighted_enc_output), dim=2))  # (batch_size, 1, out_dim)
                x = F.relu(x)

                temp_out, hidden = self.gru(x, hidden)  # seq len is 1 for each gru operation

                temp_out = self.out(temp_out) 
                output.append(temp_out)

                x = temp_out.detach()  # use prediction as next input, detach from history

        return torch.cat(tensors=output, dim=1), hidden, torch.cat(tensors=attn_weights, dim=1)

# MLP Decoder (used for Transformer Model)
class DecoderMLP(nn.Module):
    def __init__(self, args, dec_type):
        '''
        INPUTs
            args: arguments
            dec_type: type of decoder ("LR": for logistic regression of incident occurrence; "Non_LR": for speed prediction)
        '''
        super(DecoderMLP, self).__init__()
        self.args = args

        final_dim = args.out_dim * args.out_seq_len

        # for TMC speed prediction, model should output 3*out_dim results (all/truck/personal vehicle)
        if args.gt_type == "tmc" and dec_type != "LR":
            final_dim = 3*final_dim

        self.decoder = nn.Sequential(
            nn.Linear(args.in_dim*args.in_seq_len, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, final_dim)
        )

    def forward(self, x):
        '''
        INPUTs
            x: input, (batch_size, in_seq_len, in_dim)

        OUTPUTs
            result: speed prediction or incident status prediction, (batch_size, out_seq_len, out_dim)
        '''
        batch_size = x.size(0)
        result = self.decoder(x.view(batch_size, -1))  # (batch_size, out_seq_len * out_dim)

        return result.view(batch_size, self.args.out_seq_len, self.args.out_dim)


#############################
#           MODELS          #
#############################
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
            x: input, (batch_size, in_seq_len, in_dim)
            target: (batch_size, out_seq_len, out_dim), for Seq2SeqNoFact, the entries in the last dimension is speed data

        OUTPUTs
            dec_out: (batch_size, out_seq_len, out_dim)  
            dec_hidden: (num_layer, batch_size, hidden_dim)
            attn_weights: (batch_size, out_seq_len, in_seq_len)
        '''
        batch_size = x.size(0)

        # pass into encoder
        ini_hidden = self.encoder.initHidden(batch_size = batch_size)
        enc_out, enc_hidden = self.encoder(x, ini_hidden)

        # pass into decoder
        if self.args.gt_type == "tmc":
            dec_out, dec_hidden, attn_weights = self.decoder(target[..., :3].reshape(batch_size, self.args.out_seq_len+1, self.args.out_dim*3), enc_hidden ,enc_out)
        else:
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
            x: input, (batch_size, in_seq_len, num_feat_dim + incident_feat_dim)
            target: (batch_size, out_seq_len + 1, out_dim, 2), the last dimension refers to 1. speed and 2. incident status 

        OUTPUTs
            LR_out: incident status prediction, (batch_size, out_seq_len, out_dim)
            speed_pred: speed prediction, (batch_size, out_seq_len, out_dim or 3*out_dim)
            xxx_attn_weights: (batch_size, out_seq_len, in_seq_len)
        '''
        batch_size = x.size(0)

        # pass into encoder
        ini_hidden = self.encoder.initHidden(batch_size=batch_size)
        enc_out, enc_hidden = self.encoder(x, ini_hidden)

        # pass into decoder
        '''
        xxx_dec_out: (batch_size, out_seq_len, out_dim)  
        xxx_dec_hidden: (num_layer, batch_size, hidden_dim)
        xxx_attn_weights: (batch_size, out_seq_len, in_seq_len)
        '''
        if self.args.gt_type == "tmc":
            LR_out, LR_hidden, LR_attn_weights = self.LR_decoder(target[..., 3], enc_hidden, enc_out)
            rec_out, rec_hidden, rec_attn_weights = self.rec_decoder(target[..., :3].reshape(batch_size, self.args.out_seq_len+1, self.args.out_dim*3), enc_hidden, enc_out)
            nonrec_out, nonrec_hidden, nonrec_attn_weights = self.nonrec_decoder(target[..., :3].reshape(batch_size, self.args.out_seq_len+1, self.args.out_dim*3), enc_hidden, enc_out)
            
            # generate final speed prediction
            if self.args.use_gt_inc:
                    speed_pred = rec_out * (target[..., 3].repeat(1,1,3) <= 0.5) + nonrec_out * (target[..., 3].repeat(1,1,3) > 0.5)
            else:
                if self.args.use_expectation:
                    rec_weight = torch.ones(LR_out.repeat(1,1,3).size()).to(self.args.device) - LR_out.repeat(1,1,3)
                    speed_pred = rec_out * rec_weight + nonrec_out * LR_out.repeat(1,1,3) 
                else:
                    speed_pred = rec_out * (LR_out.repeat(1,1,3) <= self.args.inc_threshold) + nonrec_out * (LR_out.repeat(1,1,3) > self.args.inc_threshold)
       
        else:
            LR_out, LR_hidden, LR_attn_weights = self.LR_decoder(target[..., 1], enc_hidden, enc_out)
            rec_out, rec_hidden, rec_attn_weights = self.rec_decoder(target[..., 0], enc_hidden, enc_out)
            nonrec_out, nonrec_hidden, nonrec_attn_weights = self.nonrec_decoder(target[..., 0], enc_hidden, enc_out)

            # generate final speed prediction
            if self.args.use_gt_inc:
                    speed_pred = rec_out * (target[..., 1] <= 0.5) + nonrec_out * (target[..., 1] > 0.5)
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
            x: input, (batch_size, in_seq_len, in_dim)

        OUTPUTs
            dec_out: (batch_size, out_seq_len, out_dim or 3*out_dim)  
        '''
        enc_out = self.encoder(x)  # (batch_size, in_seq_len, in_dim)
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
            x: input, (batch_size, in_seq_len, in_dim)
            target: (batch_size, out_seq_len + 1, out_dim, 2), the last dimension refers to 1. speed and 2. incident status 

        OUTPUTs
            LR_out: incident status prediction, (batch_size, out_seq_len, out_dim)
            speed_pred: speed prediction, (batch_size, out_seq_len, out_dim or 3*out_dim)
        '''

        # pass into encoder
        enc_out = self.encoder(x)  # (batch_size, in_seq_len, in_dim)

        # pass into decoder
        '''
        xxx_dec_out: (batch_size, out_seq_len, out_dim)  
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
                    speed_pred = rec_out * (target[..., 1] <= 0.5) + nonrec_out * (target[..., 1] > 0.5)
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



