import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer

##########################
#     1. RNN Modules     #
##########################
# ----------------------------------------- TODO -----------------------------------------
# TODO - need to rewrite the MLP part (add dropout and activation)
class EncoderRNN(nn.Module):
    def __init__(self, args):
        super(EncoderRNN, self).__init__()
        self.args = args

        # self.incident_start, self.incident_end = args.incident_indices  # starting and ending indices of categorical features (incident)
        # self.incident_embedding = nn.Embedding(num_embeddings=args.incident_range, embedding_dim=args.incident_embed_dim)

        # self.input_processing = nn.Sequential(
        #         nn.Dropout(args.dropout_prob),
        #         nn.Linear(args.dim_in, 2*args.dim_hidden)
        #         )

        # For self.input_processing, one ordinary MLP layer module seems sufficient (nn.Linear + nn.BatchNorm1d + nn.ReLU + nn.Dropout)
        # In fact, it performs better than multiple layers of nn.Linear + nn.ReLU
        self.linear = nn.Linear(args.dim_in, args.dim_hidden)
        self.b_norm = nn.BatchNorm1d(num_features=args.dim_hidden, device=args.device)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout_prob)

        self.gru = nn.GRU(input_size=args.dim_hidden, hidden_size=args.dim_hidden, num_layers=args.num_layer_GRU, batch_first=True)
        # self.gru = nn.GRU(input_size=args.dim_in, hidden_size=args.dim_hidden, num_layers=args.num_layer_GRU, batch_first=True)

    def forward(self, x, hidden):
        '''
        INPUTs
            x: input, (batch_size, seq_len_in, dim_in)
            hidden: (num_layer, batch_size, dim_hidden)
        OUTPUTs
            output: (batch_size, seq_len_in, dim_hidden)
            hidden: (num_layer, batch_size, dim_hidden)
        '''
        # embedded_incident_feat = self.incident_embedding(input[:, :, self.incident_start:self.incident_end])  # (batch_size, seq_len_in, incident_feat_dim, incident_embed_dim)
        # embedded_incident_feat = torch.flatten(embedded_incident_feat, start_dim=-2)  # (batch_size, seq_len_in, incident_feat_dim * incident_embed_dim)
        # embedded_input = torch.cat((input[:self.incident_start], embedded_incident_feat, input[self.incident_end:]), dim=-1)  # (batch_size, seq_len_in, in_feat_dim)
        # output, hidden = self.gru(embedded_input, hidden)

        # processed_input = self.input_processing(x)
        processed_input = self.dropout(self.activation(torch.transpose(self.b_norm(torch.transpose(self.linear(x), 1, 2)), 1, 2)))
        output, hidden = self.gru(processed_input, hidden)
        # output, hidden = self.gru(x, hidden)

        return output, hidden

    def initHidden(self, batch_size):
        # here we supply an argument batch_size instead of using self.args.batch_size
        # because the last batch may not have full batch_size
        return torch.zeros(self.args.num_layer_GRU, batch_size, self.args.dim_hidden, device=self.args.device)
        

# RNN Decoder without Attention
class DecoderRNN(nn.Module):
    def __init__(self, args, dec_type):
        '''
        INPUTs
            args: arguments
            dec_type: type of decoder ("LR": for logistic regression of incident occurrence; "R": for regression of speed prediction)
        '''
        super(DecoderRNN, self).__init__()
        self.args = args
        self.dec_type = dec_type

        self.gru = nn.GRU(input_size=args.dim_out, hidden_size=args.dim_hidden, num_layers=args.num_layer_GRU, batch_first=True)
        
        # self.out = nn.Sequential(
        #         nn.Linear(args.dim_hidden, args.dim_out),
        #         nn.Linear(args.dim_out, args.dim_out)
        #         )

        # start from args.dim_hidden (256) -> 256 -> args.dim_out (207) -> 207

        # For self.out, adding normalization modules such as batch normalization and dropout won't help but undermine model performance 
        # Also, our input is 3-d tensor and nn.Batchnorm1d doesn't directly align with nn.Linear within nn.Sequential
        self.out = nn.Sequential(
            nn.Linear(args.dim_hidden, args.dim_hidden),
            nn.ReLU(),
            nn.Linear(args.dim_hidden, args.dim_out),
            nn.ReLU(),
            nn.Linear(args.dim_out, args.dim_out)
            )
        
    def forward(self, target, hidden):
        '''
        INPUTs
            target: (batch_size, seq_len_out, dim_out), the entries in the last dimension is either speed data or incident status
            hidden: the hidden tensor computed from encoder, (num_layer, batch_size, dim_hidden) 

        OUTPUTs
            output: (batch_size, seq_len_out, dim_out) 
            hidden: (num_layer, batch_size, dim_hidden)
        '''
        use_teacher_forcing = True if random.random() < self.args.teacher_forcing_ratio else False

        output = []
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for i in range(self.args.seq_len_out-1):
                x = target[:, i, :].unsqueeze(1)  # Teacher forcing, (batch_size, 1, dim_out)
                temp_out, hidden = self.gru(x, hidden)  # seq len is 1 for each gru operation
                output.append(self.out(temp_out))
        else:
            # Without teacher forcing: use its own predictions as the next input
            x = target[:, 0, :].unsqueeze(1)
            for i in range(self.args.seq_len_out-1):
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
        dim_out = args.dim_out

        self.attn_weight = nn.Linear(dim_out + args.dim_hidden, args.seq_len_in)
        self.attn_combine = nn.Linear(dim_out + args.dim_hidden, dim_out)

        self.gru = nn.GRU(input_size=dim_out, hidden_size=args.dim_hidden, num_layers=args.num_layer_GRU, batch_first=True)
        
        # self.out = nn.Sequential(
        #         nn.Linear(args.dim_hidden, dim_out),
        #         nn.Linear(dim_out, dim_out)
        #         )

        # start from args.dim_hidden (256) -> 256 -> args.dim_out (207) -> 207
        # For self.out, adding normalization modules such as batch normalization and dropout won't help but undermine model performance 
        # Also, our input is 3-d tensor and nn.Batchnorm1d doesn't directly align with nn.Linear within nn.Sequential
        self.out = nn.Sequential(
            nn.Linear(args.dim_hidden, args.dim_hidden),
            nn.ReLU(),
            nn.Linear(args.dim_hidden, args.dim_out),
            nn.ReLU(),
            nn.Linear(args.dim_out, args.dim_out)
            )
    

    def forward(self, target, hidden, enc_output):
        '''
        INPUTs
            target: (batch_size, seq_len_out, dim_out), the entries in the last dimension is either speed data or incident status
            hidden: the hidden tensor computed from encoder, (dec_num_layer, batch_size, dim_hidden) 
            enc_output: (batch_size, seq_len_in, dim_hidden)

        OUTPUTs
            output: (batch_size, seq_len_out, dim_out) 
            hidden: (num_layer, batch_size, dim_hidden)
            attn_weights: (batch_size, seq_len_out, seq_len_in)
        '''
        use_teacher_forcing = True if random.random() < self.args.teacher_forcing_ratio else False

        output = []
        attn_weights = []

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for i in range(self.args.seq_len_out):
                x = target[:, i, :].unsqueeze(1)  # Teacher forcing, (batch_size, 1, dim_out)

                # extract the top most hidden tensor, concat it with target input, and compute attention weights
                attn_weight = F.softmax(self.attn_weight(torch.cat(tensors=(x, hidden[-1, :, :].unsqueeze(1)), dim=2)), dim=2)  # (batch_size, 1, seq_len_in)
                attn_weights.append(attn_weight)
                
                # apply attention weights to encoder output
                weighted_enc_output = torch.bmm(attn_weight, enc_output)  # (batch_size, 1, dim_hidden)
                
                # concat weighted encoder output with target input 
                x = self.attn_combine(torch.cat(tensors=(x, weighted_enc_output), dim=2))  # (batch_size, 1, dim_out)
                x = F.relu(x)

                temp_out, hidden = self.gru(x, hidden)  # seq len is 1 for each gru operation
                output.append(self.out(temp_out))
                
        else:
            # Without teacher forcing: use its own predictions as the next input
            x = target[:, 0, :].unsqueeze(1)  # (batch_size, 1, dim_out)
            for i in range(self.args.seq_len_out):

                # extract the top most hidden tensor, concat it with target input, and compute attention weights
                attn_weight = F.softmax(self.attn_weight(torch.cat(tensors=(x, hidden[-1, :, :].unsqueeze(1)), dim=2)), dim=2)  # (batch_size, 1, seq_len_in)
                attn_weights.append(attn_weight)
                
                # apply attention weights to encoder output
                weighted_enc_output = torch.bmm(attn_weight, enc_output)  # (batch_size, 1, dim_hidden)
                
                # concat weighted encoder output with target input 
                x = self.attn_combine(torch.cat(tensors=(x, weighted_enc_output), dim=2))  # (batch_size, 1, dim_out)
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
        
        position = torch.arange(args.seq_len_in).unsqueeze(0) # (1, seq_len_in)
        div_term = torch.exp(torch.arange(0, args.dim_in, 2)*(-math.log(10000.0) / args.dim_in))
        self.pe = torch.zeros(1, args.seq_len_in, args.dim_in)
        self.pe[0, :, 0::2] = torch.sin(position * div_term)
        self.pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', self.pe)
        self.args = args


    def forward(self, x):
        '''
        INPUTs
            x: input, (batch_size, seq_len_in, dim_in)
        OUTPUTs
            output: (batch_size, seq_len_in, dim_in)
        '''
        x = x + self.pe
        return self.dropout(x)


# Transformer Encoder
class EncoderTrans(nn.Module):
    def __init__(self, args):
        super(EncoderTrans, self).__init__()
        self.args = args

        self.pos_encoder = PosEmbed(args)
        self.trans_encoder_layers = TransformerEncoderLayer(d_model=args.dim_in, nhead=args.num_head, dropout=args.dropout, batch_first=True, norm_first=True)  # layer normalization should be first, otherwise the training will be very difficult
        self.trans_encoder = TransformerEncoder(encoder_layers=self.trans_encoder_layers, nlayers=args.num_trans_layers)

        # Generates an upper-triangular matrix of -inf, with zeros on diag.
        # The masked positions (upper triangular area, excluding diag) are filled with float('-inf'). 
        # Unmasked positions (lower triangular area, including diag) are filled with float(0.0).
        self.mask = torch.triu(torch.ones(args.seq_len_in, args.seq_len_in) * float('-inf'), diagonal=1)  # size (seq_len_in, seq_len_in)

    def forward(self, x):
        '''
        INPUTs
            x: input, (batch_size, seq_len_in, dim_in)
        OUTPUTs
            output: (batch_size, seq_len_in, dim_in)
        '''
        new_x = x * math.sqrt(self.args.dim_in)
        new_x = self.pos_encoder(new_x)
        output = self.transformer_encoder(new_x, self.mask)
        return output

# ----------------------------------------- TODO -----------------------------------------
# Transformer Decoder
class DecoderTrans(nn.Module):
    def __init__(self, args):
        super(EncoderTrans, self).__init__()
        self.args = args

        self.pos_encoder = PosEmbed(args)
        self.trans_encoder_layers = TransformerEncoderLayer(d_model=args.dim_in, nhead=args.num_head, dropout=args.dropout, batch_first=True, norm_first=True)  # layer normalization should be first, otherwise the training will be very difficult
        self.trans_encoder = TransformerEncoder(encoder_layers=self.trans_encoder_layers, nlayers=args.num_trans_layers)

        # Generates an upper-triangular matrix of -inf, with zeros on diag.
        self.mask = torch.triu(torch.ones(args.seq_len_in, args.seq_len_in) * float('-inf'), diagonal=1)  # size (seq_len_in, seq_len_in)

    def forward(self, x):
        '''
        INPUTs
            x: input, (batch_size, seq_len_in, dim_in)
        OUTPUTs
            output: (batch_size, seq_len_in, dim_in)
        '''
        new_x = x * math.sqrt(self.args.dim_in)
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

        final_dim = args.dim_out * args.out_seq_len

        self.decoder = nn.Sequential(
            nn.Linear(args.dim_in*args.seq_len_in, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, final_dim)
        )

    def forward(self, x):
        '''
        INPUTs
            x: input, (batch_size, seq_len_in, dim_in)

        OUTPUTs
            result: speed prediction or incident status prediction, (batch_size, out_seq_len, dim_out)
        '''
        batch_size = x.size(0)
        result = self.decoder(x.view(batch_size, -1))  # (batch_size, out_seq_len * dim_out)

        return result.view(batch_size, self.args.out_seq_len, self.args.dim_out)



##########################
#     3. GNN Modules     #
##########################
# Spatial Module with Graph Conv Network
class SpatialModule(nn.Module):
    def __init__(self, args):
        super(SpatialModule, self).__init__()
        self.args = args
        
        self.norm_adj = nn.InstanceNorm2d(1) # normalize adjacency matrix
        # self.gcn = GCN(args, args.dim_hidden, args.dim_hidden*2, args.dim_hidden, args.adj, args.cheb_k, args.dropout)
        self.gcn = GCNConv(in_channels=args.dim_hidden, out_channels=args.dim_hidden)  

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, x): 
        '''
        INPUTs
            x: node embedding, size (batch_size, seq_len_in, num_node, dim_hidden)
            adj: adjacency matrix, size (num_node, num_node)

        OUTPUT
            result: spatial embedding, size (batch_size, seq_len_in, num_node, dim_hidden)
        '''    
        result = []
        
        for t in range(self.args.seq_len_in):
            gcn_out = self.gcn(x=x[:, t, :, :], edge_index=self.args.edge_idx).unsqueeze(1) # (batch_size, 1, num_node, dim_hidden)
            result.append(gcn_out)
        result = torch.cat(result, dim=1) # (batch_size, seq_len_in, num_node, dim_hidden)
        return result 


# Temporal Module with Transformer Encoder
class TemporalModule(nn.Module):
    def __init__(self, args):
        super(TemporalModule, self).__init__()
        
        # transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.num_node*args.dim_hidden, 
            nhead=args.num_head, 
            dim_feedforward=args.forward_expansion*args.num_node*args.dim_hidden,
            dropout=args.dropout,
            batch_first=True
            )
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=args.num_layer_Trans) 

    def forward(self, x):
        '''
        INPUT 
            x: spatial embedding, size (batch_size, seq_len_in, num_node, dim_hidden)

        OUTPUT
            result: spatial-temporal embedding, size (batch_size, seq_len_in, num_node, dim_hidden)
        '''
        batch_size, seq_len_in, num_node, dim_hidden = x.shape
        # encoder expect 2D tensor or 3D batched-tensor
        result = self.encoder(x.view(batch_size, seq_len_in, -1)) # (batch_size, seq_len_in, num_node * dim_hidden)
        return result.reshape(batch_size, seq_len_in, num_node, dim_hidden)
        

# Spatial-temporal Block (Sptial Module + Temporal Module + Skip Connection)
class STBlock(nn.Module):
    def __init__(self, args):
        super(STBlock, self).__init__()
        self.spatial_module = SpatialModule(args)
        self.temporal_module = TemporalModule(args)
        self.dropout = nn.Dropout(p=args.dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=args.dim_hidden)
    
    def forward(self, x):
        '''
        INPUT 
            x: initial node embedding, size (batch_size, seq_len_in, num_node, dim_hidden)

        OUTPUT
            result: spatial-temporal embedding (with skip connection), size (batch_size, seq_len_in, num_node, dim_hidden)
        '''
        # 1. Spatial Embedding with Spatial Module (GCN)
        spatial_embedding = self.spatial_module(x) # (batch_size, seq_len_in, num_node, dim_hidden)

        # 2. Temporal Embedding with Temporal Module (Transformer Encoder)
        spatial_temporal_embedding = self.temporal_module(spatial_embedding) # (batch_size, seq_len_in, num_node, dim_hidden)

        # 3. Skip Connection
        result = self.dropout(self.layer_norm(spatial_temporal_embedding + x))

        return result