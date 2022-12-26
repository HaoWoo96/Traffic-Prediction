import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer

##########################
#     1. RNN Modules     #
##########################
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



##################################
#     2. Transformer Modules     #
##################################
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



##########################
#     3. GNN Modules     #
##########################
# Spatial Module with Graph Conv Network
class SpatialModule(nn.Module):
    def __init__(self, args):
        super(SpatialModule, self).__init__()
        self.args = args
        
        self.norm_adj = nn.InstanceNorm2d(1) # normalize adjacency matrix
        # self.gcn = GCN(args, args.hidden_dim, args.hidden_dim*2, args.hidden_dim, args.adj, args.cheb_k, args.dropout)
        self.gcn = GCNConv(in_channels=args.hidden_dim, out_channels=args.hidden_dim)  

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, x): 
        '''
        INPUTs
            x: node embedding, size (batch_size, in_seq_len, num_node, hidden_dim)
            adj: adjacency matrix, size (num_node, num_node)

        OUTPUT
            result: spatial embedding, size (batch_size, in_seq_len, num_node, hidden_dim)
        '''    
        result = []
        
        for t in range(self.args.in_seq_len):
            gcn_out = self.gcn(x=x[:, t, :, :], edge_index=self.args.edge_idx).unsqueeze(1) # (batch_size, 1, num_node, hidden_dim)
            result.append(gcn_out)
        result = torch.cat(result, dim=1) # (batch_size, in_seq_len, num_node, hidden_dim)
        return result 


# Temporal Module with Transformer Encoder
class TemporalModule(nn.Module):
    def __init__(self, args):
        super(TemporalModule, self).__init__()
        
        # transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.num_node*args.hidden_dim, 
            nhead=args.num_head, 
            dim_feedforward=args.forward_expansion*args.num_node*args.hidden_dim,
            dropout=args.dropout,
            batch_first=True
            )
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=args.num_layer) 

    def forward(self, x):
        '''
        INPUT 
            x: spatial embedding, size (batch_size, in_seq_len, num_node, hidden_dim)

        OUTPUT
            result: spatial-temporal embedding, size (batch_size, in_seq_len, num_node, hidden_dim)
        '''
        batch_size, in_seq_len, num_node, hidden_dim = x.shape
        # encoder expect 2D tensor or 3D batched-tensor
        result = self.encoder(x.view(batch_size, in_seq_len, -1)) # (batch_size, in_seq_len, num_node * hidden_dim)
        return result.reshape(batch_size, in_seq_len, num_node, hidden_dim)
        

# Spatial-temporal Block (Sptial Module + Temporal Module + Skip Connection)
class STBlock(nn.Module):
    def __init__(self, args):
        super(STBlock, self).__init__()
        self.spatial_module = SpatialModule(args)
        self.temporal_module = TemporalModule(args)
        self.dropout = nn.Dropout(p=args.dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=args.hidden_dim)
    
    def forward(self, x):
        '''
        INPUT 
            x: initial node embedding, size (batch_size, in_seq_len, num_node, hidden_dim)

        OUTPUT
            result: spatial-temporal embedding (with skip connection), size (batch_size, in_seq_len, num_node, hidden_dim)
        '''
        # 1. Spatial Embedding with Spatial Module (GCN)
        spatial_embedding = self.spatial_module(x) # (batch_size, in_seq_len, num_node, hidden_dim)

        # 2. Temporal Embedding with Temporal Module (Transformer Encoder)
        spatial_temporal_embedding = self.temporal_module(spatial_embedding) # (batch_size, in_seq_len, num_node, hidden_dim)

        # 3. Skip Connection
        result = self.dropout(self.layer_norm(spatial_temporal_embedding + x))

        return result