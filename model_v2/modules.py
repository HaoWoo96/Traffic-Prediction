import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from torch_geometric.nn import GCNConv

from utils import *

###############################
#     Fundamental Modules     #
###############################
class InputProcessing(nn.Module):
    def __init__(self, input_size, output_size, dropout_prob, device):
        super(InputProcessing, self).__init__()

        # For self.input_processing, one ordinary MLP layer module seems sufficient (nn.Linear + nn.BatchNorm1d + nn.ReLU + nn.Dropout)
        # In fact, it performs better than multiple layers of nn.Linear + nn.ReLU
        self.linear = nn.Linear(input_size, output_size)
        self.b_norm = nn.BatchNorm1d(num_features=output_size, device=device)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        '''
        INPUT
            x: input, (batch_size, seq_len_in, input_size)
        
        OUTPUT
            output: (batch_size, seq_len_in, output_size)

        '''
        output = self.dropout(self.activation(torch.transpose(self.b_norm(torch.transpose(self.linear(x), 1, 2)), 1, 2))) # (batch_size, seq_len_in, output_size)
        return output


class OutputProcessing(nn.Module):
    def __init__(self, dim_list):
        '''
        INPUT
            dim_list: a list of integers denoting the dimensions of nn.Linear modules
        '''
        
        super(OutputProcessing, self).__init__()

        # For self.out, adding normalization modules such as batch normalization and dropout won't help but undermine model performance 
        # Also, our input is 3-d tensor and nn.Batchnorm1d doesn't directly align with nn.Linear within nn.Sequential
        module_list = []
        for i in range(len(dim_list)//2):
            module_list.append(nn.Linear(dim_list[2*i], dim_list[2*i+1]))
            if i < (len(dim_list)//2)-1:
                module_list.append(nn.ReLU())

        self.out = nn.Sequential(*module_list)
    
    def forward(self, x):
        '''
        INPUT
            x: input, (batch_size, seq_len_in, dim_in)
        
        OUTPUT
            output: (batch_size, seq_len_in, dim_out)

        '''
        return self.out(x)



##########################
#     1. RNN Modules     #
##########################
class EncoderRNN(nn.Module):
    def __init__(self, args):
        super(EncoderRNN, self).__init__()
        self.args = args

        # self.incident_start, self.incident_end = args.incident_indices  # starting and ending indices of categorical features (incident)
        # self.incident_embedding = nn.Embedding(num_embeddings=args.incident_range, embedding_dim=args.incident_embed_dim)

        # For self.input_processing, one ordinary MLP layer module seems sufficient (nn.Linear + nn.BatchNorm1d + nn.ReLU + nn.Dropout)
        # In fact, it performs better than multiple layers of nn.Linear + nn.ReLU
        self.input_processing = InputProcessing(input_size=args.dim_in, output_size=args.dim_hidden, dropout_prob=args.dropout_prob, device=args.device)

        self.gru = nn.GRU(input_size=args.dim_hidden, hidden_size=args.dim_hidden, num_layers=args.num_layer_GRU, batch_first=True)

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

        processed_input = self.input_processing(x) # (batch_size, seq_len_in, dim_hidden)
        output, hidden = self.gru(processed_input, hidden)

        return output, hidden

    def initHidden(self, batch_size):
        # here we supply an argument batch_size instead of using self.args.batch_size
        # because the last batch may not have full batch_size
        return torch.zeros(self.args.num_layer_GRU, batch_size, self.args.dim_hidden, device=self.args.device)
        

# RNN Decoder without Attention
class DecoderRNN(nn.Module):
    def __init__(self, args):
        '''
        INPUTs
            args: arguments
        '''
        super(DecoderRNN, self).__init__()
        self.args = args

        self.gru = nn.GRU(input_size=args.dim_out, hidden_size=args.dim_hidden, num_layers=args.num_layer_GRU, batch_first=True)

        # start from args.dim_hidden (256) -> 256 -> args.dim_out (207) -> 207
        # For self.out, adding normalization modules such as batch normalization and dropout won't help but undermine model performance 
        # Also, our input is 3-d tensor and nn.Batchnorm1d doesn't directly align with nn.Linear within nn.Sequential
        self.output_processing = OutputProcessing([args.dim_hidden, args.dim_hidden, args.dim_hidden, args.dim_out, args.dim_out, args.dim_out])
        
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
                output.append(self.output_processing(temp_out))
        else:
            # Without teacher forcing: use its own predictions as the next input
            x = target[:, 0, :].unsqueeze(1)
            for i in range(self.args.seq_len_out-1):
                temp_out, hidden = self.gru(x, hidden)  # seq len is 1 for each gru operation
                temp_out = self.output_processing(temp_out)
                output.append(temp_out)

                x = temp_out.detach()  # use prediction as next input, detach from history
        
        return torch.cat(tensors=output, dim=1), hidden


# RNN Decoder with Attention
class AttnDecoderRNN(nn.Module):
    def __init__(self, args):
        '''
        INPUTs
            args: arguments
        '''
        super(AttnDecoderRNN, self).__init__()
        self.args = args

        # self.dropout = nn.Dropout(self.dropout_p)
        dim_out = args.dim_out

        self.attn_weight = nn.Linear(dim_out + args.dim_hidden, args.seq_len_in)
        self.attn_combine = nn.Linear(dim_out + args.dim_hidden, dim_out)

        self.gru = nn.GRU(input_size=dim_out, hidden_size=args.dim_hidden, num_layers=args.num_layer_GRU, batch_first=True)


        # start from args.dim_hidden (256) -> 256 -> args.dim_out (207) -> 207
        # For self.out, adding normalization modules such as batch normalization and dropout won't help but undermine model performance 
        # Also, our input is 3-d tensor and nn.Batchnorm1d doesn't directly align with nn.Linear within nn.Sequential
        self.output_processing = OutputProcessing([args.dim_hidden, args.dim_hidden, args.dim_hidden, args.dim_out, args.dim_out, args.dim_out])
    

    def forward(self, target, hidden, enc_output, mode):
        '''
        INPUTs
            target: (batch_size, seq_len_out, dim_out), the entries in the last dimension is either speed data or incident status
            hidden: the hidden tensor computed from encoder, (dec_num_layer, batch_size, dim_hidden) 
            enc_output: (batch_size, seq_len_in, dim_hidden)
            mode: string of value "train" or "eval", denoting the mode to control decoder


        OUTPUTs
            output: (batch_size, seq_len_out, dim_out) 
            hidden: (num_layer, batch_size, dim_hidden)
            attn_weights: (batch_size, seq_len_out, seq_len_in)
        '''
        use_teacher_forcing = True if random.random() < self.args.teacher_forcing_ratio else False

        output = []
        attn_weights = []

        if use_teacher_forcing and mode == "train":
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
                output.append(self.output_processing(temp_out))
                
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
                temp_out = self.output_processing(temp_out)
                output.append(temp_out)

                x = temp_out.detach()  # use prediction as next input, detach from history

        return torch.cat(tensors=output, dim=1), hidden, torch.cat(tensors=attn_weights, dim=1)



##################################
#     2. Transformer Modules     #
##################################
# Positional Embedding Encoder for Transformer Encoder
class PosEmbed(nn.Module):
    def __init__(self, args, input):
        '''
        INPUTS
            args: arguments
            input: boolean value denoting whether the positional embedding is for input or not
        '''
        super(PosEmbed, self).__init__()
        self.dropout = nn.Dropout(p=args.dropout_prob)
        
        if input:
            position = torch.arange(args.seq_len_in).unsqueeze(1) # (seq_len_in, 1)
            div_term = torch.exp(torch.arange(0, args.dim_hidden, 2)*(-math.log(10000.0) / args.dim_hidden))
            self.pos_embed = torch.zeros(1, args.seq_len_in, args.dim_hidden).to(self.args.device)
            self.learnable_pos_embed = nn.Parameter(torch.zeros(args.seq_len_in, args.dim_hidden)).to(self.args.device) # learnable positional embedding 
        else:
            position = torch.arange(args.seq_len_out).unsqueeze(1) # (seq_len_out, 1)
            div_term = torch.exp(torch.arange(0, args.dim_hidden, 2)*(-math.log(10000.0) / args.dim_hidden))
            self.pos_embed = torch.zeros(1, args.seq_len_out, args.dim_hidden).to(self.args.device)
            self.learnable_pos_embed = nn.Parameter(torch.zeros(args.seq_len_out, args.dim_hidden)).to(self.args.device) # learnable positional embedding 

        self.pos_embed[0, :, 0::2] = torch.sin(position * div_term)
        self.pos_embed[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer(f"pos_embed_{'input' if input else 'output'}", self.pos_embed)

        self.args = args


    def forward(self, x):
        '''
        INPUTs
            x: PROCESSED input or target, 
                3-dimension (batch_size, seq_len_in or seq_len_out, dim_hidden) 
                or in the case of GTrans, 4-dimension (batch_size, seq_len_in or seq_len_out, num_node, dim_hidden)
        OUTPUTs
            output: (batch_size, seq_len_in or seq_len_out, dim_hidden)
        '''
        if len(x.size()) == 3:
            x = x + self.pos_embed + self.learnable_pos_embed.unsqueeze(0).unsqueeze(2)
        else:
            x = x + self.pos_embed.unsqueeze(2) + self.learnable_pos_embed.unsqueeze(0).unsqueeze(2)
        return self.dropout(x)


# Transformer Encoder
class EncoderTrans(nn.Module):
    def __init__(self, args):
        super(EncoderTrans, self).__init__()
        self.args = args

        # input processing
        self.input_processing = InputProcessing(input_size=args.dim_in, output_size=args.dim_hidden, dropout_prob=args.dropout_prob, device=args.device)

        # positional encoding
        self.pos_encoder_for_input = PosEmbed(args=args, input=True)

        # transformer encoder
        self.trans_encoder_layers = TransformerEncoderLayer(d_model=args.dim_hidden, nhead=args.num_head, dropout=args.dropout_prob, batch_first=True, norm_first=True, device=args.device)  # layer normalization should be first, otherwise the training will be very difficult
        self.trans_encoder = TransformerEncoder(encoder_layer=self.trans_encoder_layers, num_layers=args.num_layer_Trans)

        # ??? does encoder need mask ???
        # Generates an upper-triangular matrix of -inf, with zeros on diag.
        # The masked positions (upper triangular area, excluding diag) are filled with float('-inf'). 
        # Unmasked positions (lower triangular area, including diag) are filled with float(0.0).
        # self.mask = torch.triu(torch.ones(args.seq_len_in, args.seq_len_in) * float('-inf'), diagonal=1)  # size (seq_len_in, seq_len_in)

    def forward(self, x):
        '''
        INPUTs
            x: input, (batch_size, seq_len_in, dim_in)
        OUTPUTs
            output: (batch_size, seq_len_in, dim_in)
        '''
        processed_input = self.input_processing(x)  # (batch_size, seq_len_in, dim_hidden)

        # Avoid original feature embedding overshadowed by positional embedding
        # More info: # https://datascience.stackexchange.com/questions/87906/transformer-model-why-are-word-embeddings-scaled-before-adding-positional-encod/87909#87909
        processed_input = processed_input * math.sqrt(self.args.dim_hidden) 
        pos_input = self.pos_encoder_for_input(processed_input)
        # output = self.transformer_encoder(pos_input, self.mask)
        output = self.trans_encoder(pos_input)
        return output

# Transformer Decoder
class DecoderTrans(nn.Module):
    def __init__(self, args):
        super(DecoderTrans, self).__init__()
        self.args = args

        # input processing
        # here our input is the model prediction, whose feature dimension is args.dim_out
        self.input_processing = InputProcessing(input_size=args.dim_out, output_size=args.dim_hidden, dropout_prob=args.dropout_prob, device=args.device) 

        # positional embedding
        self.pos_encoder_for_output = PosEmbed(args=args, input=False)

        # transformer decoder
        self.trans_decoder_layers = TransformerDecoderLayer(d_model=args.dim_hidden, nhead=args.num_head, dropout=args.dropout_prob, batch_first=True, norm_first=True, device=args.device)  # layer normalization should be first, otherwise the training will be very difficult
        self.trans_decoder = TransformerDecoder(decoder_layer=self.trans_decoder_layers, num_layers=args.num_layer_Trans)

        # Generates an upper-triangular matrix of -inf, with zeros on diag.
        # The masked positions (upper triangular area, excluding diag) are filled with float('-inf'). 
        # Unmasked positions (lower triangular area, including diag) are filled with float(0.0).
        # tensor([[0., -inf, -inf, -inf, -inf, -inf],
        # [0., 0., -inf, -inf, -inf, -inf],
        # [0., 0., 0., -inf, -inf, -inf],
        # [0., 0., 0., 0., -inf, -inf],
        # [0., 0., 0., 0., 0., -inf],
        # [0., 0., 0., 0., 0., 0.]])
        
        self.mask = create_mask(num_row=args.seq_len_out, num_col=args.seq_len_out, device=args.device)  # size (seq_len_out, seq_len_out)
        #self.mask = torch.triu(torch.ones(args.seq_len_out, args.seq_len_out) * float('-inf'), diagonal=1).to(args.device)  

        # start from args.dim_hidden (256) -> 256 -> args.dim_out (207) -> 207
        # For self.out, adding normalization modules such as batch normalization and dropout won't help but undermine model performance 
        # Also, our input is 3-d tensor and nn.Batchnorm1d doesn't directly align with nn.Linear within nn.Sequential
        self.output_processing = OutputProcessing([args.dim_hidden, args.dim_hidden, args.dim_hidden, args.dim_out, args.dim_out, args.dim_out])

    def forward(self, target, memory, mode):
        '''
        INPUTs
            target: target, (batch_size, seq_len_out+1, dim_out)
            memory: latent representation after transformer encoder, (batch_size, seq_len_in, dim_hidden)
            mode: string of value "train" or "eval", denoting the mode to control decoder
            
        OUTPUTs
            output: (batch_size, seq_len_out, dim_out)
        '''
        use_teacher_forcing = True if random.random() < self.args.teacher_forcing_ratio else False

        if use_teacher_forcing and mode == "train":
            # Take advantage of ground truth data only if we apply teacher forcing AND we are in training mode.
            processed_tgt = self.input_processing(target[:, :self.args.seq_len_out, :])  # (batch_size, seq_len_out, dim_hidden)

            # Avoid original feature embedding overshadowed by positional embedding
            # More info: # https://datascience.stackexchange.com/questions/87906/transformer-model-why-are-word-embeddings-scaled-before-adding-positional-encod/87909#87909
            processed_tgt = processed_tgt * math.sqrt(self.args.dim_hidden) 
            pos_tgt = self.pos_encoder_for_output(processed_tgt)

            output = self.trans_decoder(tgt=pos_tgt, memory=memory, tgt_mask=self.mask)
            processed_out = self.output_processing(output)
        
        else:
            prediction = target[:, 0, :].unsqueeze(1).detach()  # (batch_size, 1, dim_out)

            for i in range(self.args.seq_len_out):
                self.input_processing(prediction)  # (batch_size, i+1, dim_hidden)
                temp_pred = self.trans_decoder(tgt=processed_tgt, memory=memory, tgt_mask=create_mask(num_row=processed_tgt.size(1), num_col=processed_tgt.size(1), device=self.args.device))  # (batch_size, i+1, dim_hidden)
                processed_pred = self.output_processing(temp_pred)  # (batch_size, i+1, dim_out)
                prediction = torch.concat((prediction, processed_pred[:, -1, :].unsqueeze(1)), 1).detach()  # (batch_size, i+2, dim_out), use prediction as next input, detach from history
            
            # processed_out = prediction[:, 1:, :]
            processed_tgt = self.dropout(self.activation(torch.transpose(self.b_norm(torch.transpose(self.linear(prediction[:, :self.args.seq_len_out, :]), 1, 2)), 1, 2)))  # (batch_size, seq_len_out, dim_hidden)

            # Avoid original feature embedding overshadowed by positional embedding
            # More info: # https://datascience.stackexchange.com/questions/87906/transformer-model-why-are-word-embeddings-scaled-before-adding-positional-encod/87909#87909
            processed_tgt = processed_tgt * math.sqrt(self.args.dim_hidden) 
            pos_tgt = self.pos_encoder_for_output(processed_tgt)

            output = self.trans_decoder(tgt=pos_tgt, memory=memory, tgt_mask=self.mask)
            processed_out = self.output_processing(output)

        return processed_out



###############################################
#     3. GTrans (GCN-Transformer) Modules     #
###############################################
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
            x: node embedding (after input processing), size (batch_size, seq_len_in, num_node, dim_hidden)
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
        
        # positional encoding
        self.pos_encoder_for_input = PosEmbed(args=args, input=True)

        # transformer encoder
        self.trans_encoder_layer = TransformerEncoderLayer(
                                        d_model=args.num_node*args.dim_hidden, 
                                        nhead=args.num_head, 
                                        dropout=args.dropout,
                                        norm_first=True,
                                        batch_first=True
                                    ) # layer normalization should be first, otherwise the training will be very difficult
        self.trans_encoder = TransformerEncoder(encoder_layer=self.trans_encoder_layers, num_layers=args.num_layer_Trans)

    def forward(self, x):
        '''
        INPUT 
            x: spatial embedding (after input processing), size (batch_size, seq_len_in, num_node, dim_hidden)

        OUTPUT
            result: spatial-temporal embedding, size (batch_size, seq_len_in, num_node, dim_hidden)
        '''
        batch_size, seq_len_in, num_node, dim_hidden = x.shape
        swapped_x = x.swapaxes(1, 2)  # (batch_size, num_node, seq_len_in, dim_hidden)

        # Avoid original feature embedding overshadowed by positional embedding
        # More info: # https://datascience.stackexchange.com/questions/87906/transformer-model-why-are-word-embeddings-scaled-before-adding-positional-encod/87909#87909
        processed_input = swapped_x * math.sqrt(self.args.dim_hidden) 
        pos_input = self.pos_encoder_for_input(processed_input)  # (batch_size, num_node, seq_len_in, dim_hidden)

        # transformer encoder expect 2D tensor or 3D batched-tensor, therefore here we merge batch and node dimension into the first dimension
        result = self.trans_encoder(pos_input.view(-1, seq_len_in, dim_hidden))  # (batch_size*num_node, seq_len_in, dim_hidden)
        result = result.reshape(batch_size, num_node, seq_len_in, dim_hidden).swapaxes(1, 2)  # (batch_size, seq_len_in, num_node, dim_hidden)
        return result
        

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
            x: node embedding after input processing, size (batch_size, seq_len_in, num_node, dim_hidden)

        OUTPUT
            result: spatial-temporal embedding (with skip connection), size (batch_size, seq_len_in, num_node, dim_hidden)
        '''
        # 1. Spatial Embedding with Spatial Module (GCN)
        spatial_embedding = self.spatial_module(x) # (batch_size, seq_len_in, num_node, dim_hidden)

        # 2. Temporal Embedding with Temporal Module (Transformer Encoder)
        spatial_temporal_embedding = self.temporal_module(spatial_embedding) # (batch_size, seq_len_in, num_node, dim_hidden)

        # 3. Skip Connection
        result = self.dropout(self.layer_norm(spatial_temporal_embedding + x))  # (batch_size, seq_len_in, num_node, dim_hidden)

        return result


# Encoder of G-Transformer
class EncoderGTrans(nn.Module):
    def __init__(self, args):
        super(EncoderGTrans, self).__init__()

        # input processing
        self.input_processing = InputProcessing(input_size=args.dim_in, output_size=args.dim_hidden, dropout_prob=args.dropout_prob, device=args.device)

        # spatial-temporal block
        self.st_blocks = nn.Sequential(*[STBlock(args=args) for _ in args.num_STBlock])
    
    def forward(self, x):
        '''
        INPUT 
            x: initial node embedding, size (batch_size, seq_len_in, num_node, dim_in)

        OUTPUT
            result: spatial-temporal embedding (with skip connection), size (batch_size, seq_len_in, num_node, dim_hidden)
        '''
        processed_input = self.input_processing(x)  # (batch_size, seq_len_in, num_node, dim_hidden)
        result = self.st_blocks(processed_input)  # (batch_size, seq_len_in, num_node, dim_hidden)

        return result


# Decoder of G-Transformer
class DecoderGTrans(nn.Module):
    def __init__(self, args):
        super(DecoderGTrans, self).__init__()
        self.args = args

        # input processing
        # here our input is the model prediction, whose feature dimension is args.dim_out
        self.input_processing = InputProcessing(input_size=args.dim_out, output_size=args.dim_hidden, dropout_prob=args.dropout_prob, device=args.device) 

        # positional embedding
        self.pos_encoder_for_output = PosEmbed(args=args, input=False)

        # transformer decoder
        self.trans_decoder_layers = TransformerDecoderLayer(d_model=args.dim_hidden, 
                                        nhead=args.num_head, 
                                        dropout=args.dropout_prob, 
                                        batch_first=True, 
                                        norm_first=True, 
                                        device=args.device
                                    )  # layer normalization should be first, otherwise the training will be very difficult
        self.trans_decoder = TransformerDecoder(decoder_layer=self.trans_decoder_layers, num_layers=args.num_layer_Trans)

        # Generates an upper-triangular matrix of -inf, with zeros on diag.
        # The masked positions (upper triangular area, excluding diag) are filled with float('-inf'). 
        # Unmasked positions (lower triangular area, including diag) are filled with float(0.0).
        # tensor([[0., -inf, -inf, -inf, -inf, -inf],
        # [0., 0., -inf, -inf, -inf, -inf],
        # [0., 0., 0., -inf, -inf, -inf],
        # [0., 0., 0., 0., -inf, -inf],
        # [0., 0., 0., 0., 0., -inf],
        # [0., 0., 0., 0., 0., 0.]])
        
        self.mask = create_mask(num_row=args.seq_len_out, num_col=args.seq_len_out, device=args.device)  # size (seq_len_out, seq_len_out)
        #self.mask = torch.triu(torch.ones(args.seq_len_out, args.seq_len_out) * float('-inf'), diagonal=1).to(args.device)  

        # start from args.dim_hidden (28) -> 28 -> args.dim_out (1) -> 1
        # For self.out, adding normalization modules such as batch normalization and dropout won't help but undermine model performance 
        # Also, our input is 3-d tensor and nn.Batchnorm1d doesn't directly align with nn.Linear within nn.Sequential
        self.output_processing = OutputProcessing([args.dim_hidden, args.dim_hidden, args.dim_hidden, 16, 16, args.dim_out])

    def forward(self, target, memory, mode):
        '''
        INPUTs
            target: target, (batch_size, seq_len_out+1, num_node, dim_out)
            memory: latent representation after transformer encoder, (batch_size, seq_len_in, num_node, dim_hidden)
            mode: string of value "train" or "eval", denoting the mode to control decoder
            
        OUTPUTs
            output: (batch_size, seq_len_out, num_node, dim_out)
        '''
        batch_size, seq_len_out, num_node, dim_out = target.size()
        seq_len_out -= 1
        dim_hidden = memory.size(-1)
        use_teacher_forcing = True if random.random() < self.args.teacher_forcing_ratio else False

        if use_teacher_forcing and mode == "train":
            # Take advantage of ground truth data only if we apply teacher forcing AND we are in training mode.
            processed_tgt = self.input_processing(target[:, :seq_len_out, :, :])  # (batch_size, seq_len_out, num_node, dim_hidden)

            # Avoid original feature embedding overshadowed by positional embedding
            # More info: # https://datascience.stackexchange.com/questions/87906/transformer-model-why-are-word-embeddings-scaled-before-adding-positional-encod/87909#87909
            processed_tgt = processed_tgt * math.sqrt(self.args.dim_hidden) 
            pos_tgt = self.pos_encoder_for_output(processed_tgt)

            output = self.trans_decoder(tgt=pos_tgt.swapaxes(1,2).view(-1, num_node, dim_hidden), memory=memory, tgt_mask=self.mask)  # (batch_size*num_node, seq_len_out, dim_hidden)
            output = output.swapaxes(1,2).reshape(batch_size, seq_len_out, num_node, dim_hidden)  # (batch_size, seq_len_out, num_node, dim_hidden)
            processed_out = self.output_processing(output)
        
        else:
            prediction = target[:, 0, :, :].unsqueeze(1).detach()  # (batch_size, 1, num_node, dim_out)

            for i in range(self.args.seq_len_out):
                processed_tgt = self.input_processing(prediction)  # (batch_size, i+1, num_node, dim_hidden)
                temp_pred = self.trans_decoder(tgt=processed_tgt.swapaxes(1,2).view(-1, self.args.num_node, self.args.dim_hidden), memory=memory, tgt_mask=create_mask(num_row=processed_tgt.size(1), num_col=processed_tgt.size(1), device=self.args.device))  # (batch_size*num_node, i+1, dim_hidden)
                temp_pred = temp_pred.swapaxes(1,2).reshape(batch_size, seq_len_out, num_node, dim_hidden)  # (batch_size, i+1, num_node, dim_hidden)
                processed_pred = self.output_processing(temp_pred)  # (batch_size, i+1, num_node, dim_out)
                prediction = torch.concat((prediction, processed_pred[:, -1, :].unsqueeze(1)), 1).detach()  # (batch_size, i+2, num_node, dim_out), use prediction as next input, detach from history
            
            # processed_out = prediction[:, 1:, :]
            processed_tgt = self.input_processing(prediction[:, :self.args.seq_len_out, :]) # (batch_size, seq_len_out, num_node, dim_hidden)

            # Avoid original feature embedding overshadowed by positional embedding
            # More info: # https://datascience.stackexchange.com/questions/87906/transformer-model-why-are-word-embeddings-scaled-before-adding-positional-encod/87909#87909
            processed_tgt = processed_tgt * math.sqrt(self.args.dim_hidden) 
            pos_tgt = self.pos_encoder_for_output(processed_tgt)

            output = self.trans_decoder(tgt=pos_tgt.swapaxes(1,2).view(-1, num_node, dim_hidden), memory=memory, tgt_mask=self.mask)  # (batch_size*num_node, seq_len_out, dim_hidden)
            output = output.swapaxes(1,2).reshape(batch_size, seq_len_out, num_node, dim_hidden)  # (batch_size, seq_len_out, num_node, dim_hidden)
            processed_out = self.output_processing(output)

        return processed_out