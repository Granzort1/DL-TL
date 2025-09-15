import os
import glob
import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import matplotlib.dates as mdates
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from matplotlib import font_manager, rc

# Set the font properties
pd.set_option('display.max_columns', None)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.unicode_minus'] = False  # 한글 폰트 사용시 마이너스 기호 깨짐 방지

#### Set seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available)
print(torch.version.cuda)
print(device)

class GLU(nn.Module):
 
    def __init__(self, input_size):
        super().__init__()
        
        # Input
        self.a = nn.Linear(input_size, input_size)

        # Gate
        self.sigmoid = nn.Sigmoid()
        self.b = nn.Linear(input_size, input_size)
        
    def forward(self, x):
        """
        Args:
            x (torch.tensor): tensor passing through the gate
        """
        gate = self.sigmoid(self.b(x))
        x = self.a(x)
        
        return torch.mul(gate, x)

class TemporalLayer(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module


    def forward(self, x):
    
        t, n = x.size(0), x.size(1)
        x = x.reshape(t * n, -1)
        x = self.module(x)
        x = x.reshape(t, n, x.size(-1))

        return x

class GatedResidualNetwork(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, dropout, context_size=None, is_temporal=True):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.is_temporal = is_temporal
        
        if self.is_temporal:
            if self.input_size != self.output_size:
                self.skip_layer = TemporalLayer(nn.Linear(self.input_size, self.output_size))

            # Context vector c
            if self.context_size != None:
                self.c = TemporalLayer(nn.Linear(self.context_size, self.hidden_size, bias=False))

            # Dense & ELU
            self.dense1 = TemporalLayer(nn.Linear(self.input_size, self.hidden_size))
            self.elu = nn.ELU()

            # Dense & Dropout
            self.dense2 = TemporalLayer(nn.Linear(self.hidden_size,  self.output_size))
            self.dropout = nn.Dropout(self.dropout)

            # Gate, Add & Norm
            self.gate = TemporalLayer(GLU(self.output_size))
            self.layer_norm = TemporalLayer(nn.BatchNorm1d(self.output_size))

        else:
            if self.input_size != self.output_size:
                self.skip_layer = nn.Linear(self.input_size, self.output_size)

            # Context vector c
            if self.context_size != None:
                self.c = nn.Linear(self.context_size, self.hidden_size, bias=False)

            # Dense & ELU
            self.dense1 = nn.Linear(self.input_size, self.hidden_size)
            self.elu = nn.ELU()

            # Dense & Dropout
            self.dense2 = nn.Linear(self.hidden_size,  self.output_size)
            self.dropout = nn.Dropout(self.dropout)

            # Gate, Add & Norm
            self.gate = GLU(self.output_size)
            self.layer_norm = nn.BatchNorm1d(self.output_size)


    def forward(self, x, c=None):

        if self.input_size!=self.output_size:
            a = self.skip_layer(x)
        else:
            a = x
        
        x = self.dense1(x)

        if c != None:
            c = self.c(c.unsqueeze(1))
            x += c

        eta_2 = self.elu(x)
        
        eta_1 = self.dense2(eta_2)
        eta_1 = self.dropout(eta_1)

        gate = self.gate(eta_1)
        gate += a
        ######################################Apply conditional BatchNorm based on batch size############################
        ##############################################################################################################
        ##############################################################################################################
        ##############################################################################################################
        if gate.size(0) > 1:  # Only apply BatchNorm if batch size is greater than 1
            x = self.layer_norm(gate)
        else:
            x = gate  # Skip normalization if batch size is too small
        
        return x

class VariableSelectionNetwork(nn.Module):
  
    def __init__(self, input_size, output_size, hidden_size, dropout, context_size=None, is_temporal=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.context_size = context_size
        self.is_temporal = is_temporal
       
        self.flattened_inputs = GatedResidualNetwork(self.output_size*self.input_size, 
                                                     self.hidden_size, self.output_size, 
                                                     self.dropout, self.context_size, 
                                                     self.is_temporal)
        
        self.transformed_inputs = nn.ModuleList(
            [GatedResidualNetwork(
                self.input_size, self.hidden_size, self.hidden_size, 
                self.dropout, self.context_size, self.is_temporal) for i in range(self.output_size)])

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, embedding, context=None):

        # Generation of variable selection weights
        sparse_weights = self.flattened_inputs(embedding, context)
        if self.is_temporal:
            sparse_weights = self.softmax(sparse_weights).unsqueeze(2)
        else:
            sparse_weights = self.softmax(sparse_weights).unsqueeze(1)

        # Additional non-linear processing for each feature vector
        transformed_embeddings = torch.stack(
            [self.transformed_inputs[i](embedding[
                Ellipsis, i*self.input_size:(i+1)*self.input_size]) for i in range(self.output_size)], axis=-1)

        # Processed features are weighted by their corresponding weights and combined
        combined = transformed_embeddings*sparse_weights
        combined = combined.sum(axis=-1)

        return combined, sparse_weights

class ScaledDotProductAttention(nn.Module):    #MUTLI-HEAD SELF-ATTENTION
  
    def __init__(self, dropout=0.0): 
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)
        

    def forward(self, query, key, value, mask=None):

        d_k = key.shape[-1]
        scaling_factor = torch.sqrt(torch.tensor(d_k).to(torch.float32))

        scaled_dot_product = torch.matmul(query, key.permute(0,2,1)) / scaling_factor 
        if mask != None:
            scaled_dot_product = scaled_dot_product.masked_fill(mask == 0, -1e9)
        attention = self.softmax(scaled_dot_product)
        attention = self.dropout(attention)
        output = torch.matmul(attention, value)

        return output, attention

class InterpretableMultiHeadAttention(nn.Module):
   
    def __init__(self, num_attention_heads, hidden_size, dropout=0.0):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        self.qs = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=False) for i in range(self.num_attention_heads)])
        self.ks = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size, bias=False) for i in range(self.num_attention_heads)])

        vs_layer = nn.Linear(self.hidden_size, self.hidden_size, bias=False) # Value is shared for improved interpretability
        self.vs = nn.ModuleList([vs_layer for i in range(self.num_attention_heads)])

        self.attention = ScaledDotProductAttention()
        self.linear = nn.Linear(self.hidden_size, self.hidden_size, bias=False)


    def forward(self, query, key, value, mask=None):
        
        batch_size, tgt_len, embed_dim = query.shape
        head_dim = embed_dim // self.num_attention_heads

        # Now we iterate over each head to calculate outputs and attention
        heads = []
        attentions = []

        for i in range(self.num_attention_heads):
            q_i = self.qs[i](query)
            k_i = self.ks[i](key)
            v_i = self.vs[i](value)

            # Reshape q, k, v for multihead attention
            q_i = query.reshape(batch_size, tgt_len, self.num_attention_heads, head_dim).transpose(1,2).reshape(batch_size*self.num_attention_heads, tgt_len, head_dim)
            k_i = key.reshape(batch_size, tgt_len, self.num_attention_heads, head_dim).transpose(1,2).reshape(batch_size*self.num_attention_heads, tgt_len, head_dim)
            v_i = value.reshape(batch_size, tgt_len, self.num_attention_heads, head_dim).transpose(1,2).reshape(batch_size*self.num_attention_heads, tgt_len, head_dim)

            head, attention = self.attention(q_i, k_i, v_i, mask)

            # Revert to original target shape
            head = head.reshape(batch_size, self.num_attention_heads, tgt_len, head_dim).transpose(1,2).reshape(-1, tgt_len, self.num_attention_heads*head_dim)
            head_dropout = self.dropout(head)
            heads.append(head_dropout)
            attentions.append(attention)

        # Output the results
        if self.num_attention_heads > 1:
            heads = torch.stack(heads, dim=2) #.reshape(batch_size, tgt_len, -1, self.hidden_size)
            outputs = torch.mean(heads, dim=2)
        else:
            outputs = head

        attentions = torch.stack(attentions, dim=2)
        attention = torch.mean(attentions, dim=2)
        
        outputs = self.linear(outputs)
        outputs = self.dropout(outputs)

        return outputs, attention

class QuantileLoss(nn.Module):

    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
    
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)

        losses = []
        for i, q in enumerate(self.quantiles):  
            errors = target - preds[:, :, i]
            losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(1))
        
        loss = torch.mean(torch.sum(torch.cat(losses, dim=2), dim=2))

        return loss


class TFT(nn.Module):
  
    def __init__(
        self, 
        quantiles, 
        dropout,
        device,
        hidden_layer_size,
        num_lstm_layers,
        embedding_dim,
        encoder_steps,
        num_attention_heads,
        col_to_idx,
        static_covariates,
        time_dependent_categorical,
        time_dependent_continuous,
        category_counts,
        known_time_dependent,
        observed_time_dependent,
    ):

        super().__init__()

        # Inputs
        self.col_to_idx = col_to_idx
        self.static_covariates = static_covariates
        self.time_dependent_categorical = time_dependent_categorical
        self.time_dependent_continuous = time_dependent_continuous
        self.category_counts = category_counts
        self.known_time_dependent = known_time_dependent
        self.observed_time_dependent = observed_time_dependent
        self.time_dependent = self.known_time_dependent+self.observed_time_dependent

        # Architecture
        self.encoder_steps = encoder_steps
        self.hidden_size = hidden_layer_size
        self.num_lstm_layers = num_lstm_layers
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads

        # Outputs
        self.quantiles = quantiles

        # Other
        self.device = device

        # Prepare embeddings for the static covariates and static context vectors
        self.static_embeddings = nn.ModuleDict({col: nn.Embedding(self.category_counts[col], self.embedding_dim).to(self.device) for col in self.static_covariates}) 
        self.static_variable_selection = VariableSelectionNetwork(self.embedding_dim, len(self.static_covariates), self.hidden_size, self.dropout, is_temporal=False) 

        self.static_context_variable_selection = GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout, is_temporal=False)
        self.static_context_enrichment = GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout, is_temporal=False)
        self.static_context_state_h = GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout, is_temporal=False)
        self.static_context_state_c = GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout, is_temporal=False)
        
        # Prepare embeddings and linear transformations for time dependent variables
        self.temporal_cat_embeddings = nn.ModuleDict({col: TemporalLayer(nn.Embedding(self.category_counts[col], self.embedding_dim)).to(self.device) for col in self.time_dependent_categorical})
        self.temporal_real_transformations = nn.ModuleDict({col: TemporalLayer(nn.Linear(1, self.embedding_dim)).to(self.device) for col in self.time_dependent_continuous})

        # Variable selection and encoder for past inputs
        self.past_variable_selection = VariableSelectionNetwork(self.embedding_dim, len(self.time_dependent), self.hidden_size, self.dropout, context_size=self.hidden_size)

        # Variable selection and decoder for known future inputs
        self.future_variable_selection = VariableSelectionNetwork(self.embedding_dim, len([col for col in self.time_dependent if col not in self.observed_time_dependent]), 
                                                                  self.hidden_size, self.dropout, context_size=self.hidden_size)

        # LSTM encoder and decoder
        self.lstm_encoder = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_lstm_layers, dropout=self.dropout)
        self.lstm_decoder = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_lstm_layers, dropout=self.dropout)

        # Gated skip connection and normalization
        self.gated_skip_connection = TemporalLayer(GLU(self.hidden_size))
        self.add_norm = TemporalLayer(nn.BatchNorm1d(self.hidden_size))

        # Temporal Fusion Decoder

        # Static enrichment layer
        self.static_enrichment = GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout, self.hidden_size)
        
        # Temporal Self-attention layer
        self.multihead_attn = InterpretableMultiHeadAttention(self.num_attention_heads, self.hidden_size)
        self.attention_gated_skip_connection = TemporalLayer(GLU(self.hidden_size))
        self.attention_add_norm = TemporalLayer(nn.BatchNorm1d(self.hidden_size, self.hidden_size))

        # Position-wise feed-forward layer
        self.position_wise_feed_forward = GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout)

        # Output layer
        self.output_gated_skip_connection = TemporalLayer(GLU(self.hidden_size))
        self.output_add_norm = TemporalLayer(nn.BatchNorm1d(self.hidden_size, self.hidden_size))

        self.output = TemporalLayer(nn.Linear(self.hidden_size, len(self.quantiles)))
        
  
    def define_static_covariate_encoders(self, x):
        embedding_vectors = [self.static_embeddings[col](x[:, 0, self.col_to_idx[col]].long().to(self.device)) for col in self.static_covariates]
        static_embedding = torch.cat(embedding_vectors, dim=1)
        static_encoder, static_weights = self.static_variable_selection(static_embedding)

        # Static context vectors
        static_context_s = self.static_context_variable_selection(static_encoder) # Context for temporal variable selection
        static_context_e = self.static_context_enrichment(static_encoder) # Context for static enrichment layer
        static_context_h = self.static_context_state_h(static_encoder) # Context for local processing of temporal features (encoder/decoder)
        static_context_c = self.static_context_state_c(static_encoder) # Context for local processing of temporal features (encoder/decoder)

        return static_encoder, static_weights, static_context_s, static_context_e, static_context_h, static_context_c

    def define_past_inputs_encoder(self, x, context):
        # print("Time dependent categorical variables (past):", self.time_dependent_categorical)
        # print("Time dependent continuous variables (past):", self.time_dependent_continuous)

        # Only concatenate embeddings if there are any time-dependent categorical variables
        if self.time_dependent_categorical:
            embedding_vectors = torch.cat([self.temporal_cat_embeddings[col](x[:, :, self.col_to_idx[col]].long()) for col in self.time_dependent_categorical], dim=2)
        else:
            embedding_vectors = torch.empty((x.size(0), x.size(1), 0)).to(self.device)

        transformation_vectors = torch.cat([self.temporal_real_transformations[col](x[:, :, self.col_to_idx[col]]) for col in self.time_dependent_continuous], dim=2)

        past_inputs = torch.cat([embedding_vectors, transformation_vectors], dim=2)
        past_encoder, past_weights = self.past_variable_selection(past_inputs, context)

        return past_encoder.transpose(0, 1), past_weights


    def define_known_future_inputs_decoder(self, x, context):
        # print("Time dependent continuous variables:", self.time_dependent_continuous)
        # print("Observed time dependent variables:", self.observed_time_dependent)

        # Only concatenate embeddings if there are any time-dependent categorical variables
        if self.time_dependent_categorical:
            embedding_vectors = torch.cat([self.temporal_cat_embeddings[col](x[:, :, self.col_to_idx[col]].long()) 
                                        for col in self.time_dependent_categorical if col not in self.observed_time_dependent], dim=2)
        else:
            embedding_vectors = torch.empty((x.size(0), x.size(1), 0)).to(self.device)

        transformation_vectors = torch.cat([self.temporal_real_transformations[col](x[:, :, self.col_to_idx[col]]) 
                                            for col in self.time_dependent_continuous if col not in self.observed_time_dependent], dim=2)

        future_inputs = torch.cat([embedding_vectors, transformation_vectors], dim=2)
        future_decoder, future_weights = self.future_variable_selection(future_inputs, context)

        return future_decoder.transpose(0, 1), future_weights

    def define_lstm_encoder(self, x, static_context_h, static_context_c):
        output, (state_h, state_c) = self.lstm_encoder(x, (static_context_h.unsqueeze(0).repeat(self.num_lstm_layers,1,1), 
                                                           static_context_c.unsqueeze(0).repeat(self.num_lstm_layers,1,1)))
        
        return output, state_h, state_c


    def define_lstm_decoder(self, x, state_h, state_c):
        output, (_, _) = self.lstm_decoder(x, (state_h.unsqueeze(0).repeat(self.num_lstm_layers,1,1), 
                                               state_c.unsqueeze(0).repeat(self.num_lstm_layers,1,1)))
        
        return output

    
    def get_mask(self, attention_inputs):
        #mask = torch.cumsum(torch.eye(attention_inputs.shape[1]*self.num_attention_heads, attention_inputs.shape[0]), dim=1)
        mask = torch.cumsum(torch.eye(attention_inputs.shape[0]*self.num_attention_heads, attention_inputs.shape[1]), dim=1)

        return mask.unsqueeze(2).to(self.device)

    def forward(self, x):

        # Static variable selection and static covariate encoders
        static_encoder, static_weights, static_context_s, static_context_e, static_context_h, static_context_c = self.define_static_covariate_encoders(x)

        # Past input variable selection and LSTM encoder
        past_encoder, past_weights = self.define_past_inputs_encoder(x[:, :self.encoder_steps, :].float().to(self.device), static_context_s)

        # Known future inputs variable selection and LSTM decoder
        future_decoder, future_weights = self.define_known_future_inputs_decoder(x[:, self.encoder_steps:, :].float().to(self.device), static_context_s)

        # Pass output from variable selection through LSTM encoder and decoder
        encoder_output, state_h, state_c = self.define_lstm_encoder(past_encoder, static_context_h, static_context_c)
        decoder_output = self.define_lstm_decoder(future_decoder, static_context_h, static_context_c)

        # Gated skip connection before moving into the Temporal Fusion Decoder
        variable_selection_outputs = torch.cat([past_encoder, future_decoder], dim=0)
        lstm_outputs = torch.cat([encoder_output, decoder_output], dim=0)
        gated_outputs = self.gated_skip_connection(lstm_outputs)
        temporal_feature_outputs = self.add_norm(variable_selection_outputs.add(gated_outputs))
        temporal_feature_outputs = temporal_feature_outputs.transpose(0, 1)

        # Temporal Fusion Decoder
        # Static enrcihment layer
        static_enrichment_outputs = self.static_enrichment(temporal_feature_outputs, static_context_e)

        # Temporal Self-attention layer
        mask = self.get_mask(static_enrichment_outputs)
        multihead_outputs, multihead_attention = self.multihead_attn(static_enrichment_outputs, static_enrichment_outputs, static_enrichment_outputs, mask=mask)
        
        attention_gated_outputs = self.attention_gated_skip_connection(multihead_outputs)
        attention_outputs = self.attention_add_norm(attention_gated_outputs.add(static_enrichment_outputs))

        # Position-wise feed-forward layer
        temporal_fusion_decoder_outputs = self.position_wise_feed_forward(attention_outputs)

        # Output layer
        gate_outputs = self.output_gated_skip_connection(temporal_fusion_decoder_outputs)
        norm_outputs = self.output_add_norm(gate_outputs.add(temporal_feature_outputs))

        output = self.output(norm_outputs[:, self.encoder_steps:, :])
        
        attention_weights = {
            'multihead_attention': multihead_attention,
            'static_weights': static_weights[Ellipsis, 0],
            'past_weights': past_weights[Ellipsis, 0, :],
            'future_weights': future_weights[Ellipsis, 0, :]
        }
        return  output

def round_columns(df, keyword, decimals):
    for col in df.columns:
        if keyword in col.split('_')[0]:  # Check up to the first underscore
            df[col] = np.round(df[col].astype(float), decimals)
    return df

# Class to create a PyTorch Dataset for time series
class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, flags, device):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)
        self.flags = torch.tensor(flags, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.flags[index]

# Function to create sequences for training
def create_sequences(data, target, seq_len, out_seq, target_values, flags=None):
    # Ensure the index is a datetime index
    period = list(data.index)
    targets = list(data.columns)

    start_ = period.index(data.loc[~data[target].isna(), :].index[0])
    end_ = period.index(data.loc[~data[target].isna(), :].index[-1])
    
    d = np.array(data)[start_: end_ + 1, :]
    t = np.array(target_values)[start_: end_ + 1]  # Log-transformed target values
    f = np.array(flags)[start_: end_ + 1, :] if flags is not None else None

    xxs, yys, flags_list = [], [], []
    for i in range(len(d) - seq_len - out_seq + 1):
        xx = d[i:(i + seq_len + out_seq), :]
        yy = t[i + seq_len + out_seq - 1]
        flag = f[i + seq_len + out_seq - 1, flags.columns.get_loc(target)] if flags is not None else 0

        xxs.append(xx)
        yys.append(yy)
        if flags is not None:
            flags_list.append(flag)

    xxs_array = np.array(xxs)
    yys_array = np.array(yys).reshape(-1, 1)
    flags_array = np.array(flags_list).reshape(-1, 1) if flags is not None else None
    
    return xxs_array, yys_array, flags_array

def train_model_with_flag_masking(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, EPOCHS, random_seed=42, patience=50):
    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    best_validation_loss = float('inf')
    no_improve = 0                                                
    last_lr = optimizer.param_groups[0]['lr']     
                    
    for epoch in range(1, EPOCHS + 1):  # Epoch starts from 1 for readability
        # Training loop
        model.train()
        total_train_loss = 0
        count_actual_values_train = 0
        for batch_idx, (train_inputs, train_targets, train_flags) in enumerate(train_dataloader):
            optimizer.zero_grad()
            train_outputs = model(train_inputs)

            # Element-wise masking for training
            mask = train_flags == 0
            mask_expanded = mask.unsqueeze(-1).expand_as(train_outputs)
            
            train_outputs_masked = train_outputs * mask_expanded.float()
            train_targets_masked = train_targets * mask.float()

            # If valid, calculate the training loss
            if mask_expanded.sum() > 0:
                train_loss = criterion(train_outputs_masked, train_targets_masked)
                train_loss.backward()
                optimizer.step()

                # Accumulate total training loss for logging purposes
                total_train_loss += train_loss.item()
                count_actual_values_train += 1

        # Calculate average training loss
        avg_train_loss = total_train_loss / count_actual_values_train if count_actual_values_train > 0 else float('inf')

        # Validation loop
        model.eval()
        validation_loss = 0
        count_actual_values = 0
        with torch.no_grad():
            for batch_idx, (valid_inputs, valid_targets, valid_flags) in enumerate(valid_dataloader):
                valid_outputs = model(valid_inputs)

                # Element-wise masking for validation
                mask = valid_flags == 0
                mask_expanded = mask.unsqueeze(-1).expand_as(valid_outputs)
                
                valid_outputs_masked = valid_outputs * mask_expanded.float()
                valid_targets_masked = valid_targets * mask.float()

                # If valid, calculate the validation loss
                if mask_expanded.sum() > 0:
                    valid_loss = criterion(valid_outputs_masked, valid_targets_masked)
                    validation_loss += valid_loss.item()
                    count_actual_values += 1

        # Calculate average validation loss
        if count_actual_values > 0:
            validation_loss /= count_actual_values
        
        # Print progress every epoch
        current_lr = optimizer.param_groups[0]['lr']
        if epoch % 5 == 0 or current_lr != last_lr:
            print(f"Epoch {epoch}/{EPOCHS}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {validation_loss:.4f}, LR: {current_lr:.6f}")
        
        # Check if learning rate has changed
        if current_lr != last_lr:
            print(f"Learning rate changed to {current_lr:.6f} at epoch {epoch}")
            last_lr = current_lr

        scheduler.step(validation_loss)

        # Early stopping check
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch}, best validation loss: {best_validation_loss:.4f}')
            return model

    return model
    
def log10_1p(x):
    return np.log10(1 + x) 

def filter_columns(df, variables):
    return df[[col for col in df.columns if any(var in col for var in variables)]]

def kalman_filter_interpolation(df):

    df_copy = df.copy()
    interpolation_flags = pd.DataFrame(0, index=df_copy.index, columns=df_copy.columns)

    for column in df_copy.columns:
        values = df_copy[column].values
        if np.isnan(values).any():
            masked_values = np.ma.masked_invalid(values)
            kf = KalmanFilter(
                initial_state_mean=np.nanmean(values),
                n_dim_obs=1,
                transition_matrices=[1],
                observation_matrices=[1],
                transition_covariance=np.eye(1),
                observation_covariance=np.eye(1)
            )
            kf = kf.em(masked_values, n_iter=10)
            filled_values, _ = kf.smooth(masked_values)
            df_copy[column] = np.where(np.isnan(values), filled_values.flatten(), values)
            
            # Mark interpolated values
            interpolation_flags[column] = np.where(np.isnan(values), 1, 0)
            # print(f"Interpolated rows for column '{column}': {np.where(np.isnan(values))[0]}")
            # print(f"Flags for column '{column}': {interpolation_flags[column].values}")

    return df_copy, interpolation_flags

    
## Change path
river_data_info = {
   'NAK': {
       'locations': {
           'SITE': '/PATH/TO/YOUR/CSV/FILE/SJB_nak_total.csv', 
       }
   },
}

# Global parameters
Input_Variables = ['Cyanocell', 'WT', 'Chla', 'TN', 'TP', 'WL', 'Discharge','Temp', 'Prec','Forecast']
quantiles = [0.2, 0.5, 0.8]
batch_size = 32
Input_Sequence = 2
Output_Sequence = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 반복문을 통해 각 데이터프레임에 대해 모델 훈련 및 저장
Sc_tr_x = []
Sc_tr_y = []
Sc_vl_x = []
Sc_vl_y = []
Sc_te_x = []
Sc_te_y = []
Sc_tr_flags = []
Sc_vl_flags = []
Sc_te_flags = []

# Loop over each river and its locations
for river_name, river_info in river_data_info.items():
  
    for location_name, filepath in river_info['locations'].items():
        print(f"Starting training for River: {river_name}, Location: {location_name}")

        df = pd.read_csv(filepath, encoding = 'cp949', index_col='Date')
        
        df.index = pd.to_datetime(df.index).normalize()

        interpolated_data, interpolation_flags = kalman_filter_interpolation(df)
        # Concatenate interpolated data and flags for processing
        temp_df = filter_columns(interpolated_data, Input_Variables)
        flags_df = filter_columns(interpolation_flags, Input_Variables)
        temp_df2 = temp_df.copy()

        temp_df2.loc[:, 'Month'] = temp_df2.index.month
        temp_df2.loc[:, 'season'] = temp_df2['Month'].apply(lambda x: '여름' if x in [6, 7, 8, 9, 10] else '이외')
        temp_df2=temp_df2.drop(columns=['Month'])
        transform_df = temp_df2.copy()
        
        # Convert categprocal variables
        categorical_scalers = {}
        categorical_columns = ['season']
        category_counts = {'season':2}
        
        for col in categorical_columns:
            srs = transform_df[col].apply(str) 
            categorical_scalers[col] = LabelEncoder().fit(srs.values)
            transform_df[col] = categorical_scalers[col].transform(transform_df[col].apply(str))

        # Create input
        new_input_columns = []
        for col in temp_df2.columns:
            if col == 'date':
                continue
            new_input_columns.append(col)
            
        # select where you want to put your inputs in. If not categoricals, variables go to time dependent continuous with the unknown variables.    
        static_covariates = ['season']
        time_dependent_categorical = []
        time_dependent_continuous = []

        for col in new_input_columns:
            if col in static_covariates or col in time_dependent_categorical:
                continue
            time_dependent_continuous.append(col)
        
        # Select known time-dependent variables as columns containing 'Forecast'
        known_time_dependent = [col for col in new_input_columns if 'Forecast' in col or col == 'Month']
        
        # Select observed time-dependent variables
        observed_time_dependent = []
        for col in new_input_columns:
            if col in static_covariates or col in known_time_dependent:
                continue
            observed_time_dependent.append(col)
        
        col_to_idx = {col: idx for idx, col in enumerate(new_input_columns)}  
        
        # Select output variable
        out_var = [col for col in temp_df2.columns if 'Cyanocell' in col][0]
        Output_Variables = [out_var]
        target_column = out_var
        X = transform_df[new_input_columns]
        y = transform_df[Output_Variables]
        
        # Create dataset
        start_date = transform_df.index.min()
        X.index = X.index.normalize()
        y.index = y.index.normalize()

        X_train = X.truncate(before=start_date, after='2021-12-31')
        y_train = y.truncate(before=start_date, after='2021-12-31')
        flags_train = flags_df.truncate(before=start_date, after='2021-12-31')

        X_valid = X.truncate(before='2022-01-01', after='2022-12-31')
        y_valid = y.truncate(before='2022-01-01', after='2022-12-31')
        flags_valid = flags_df.truncate(before='2022-01-01', after='2022-12-31')

        X_test = X.truncate(before='2023-01-01', after='2023-12-31')
        y_test = y.truncate(before='2023-01-01', after='2023-12-31')
        flags_test = flags_df.truncate(before='2023-01-01', after='2023-12-31')

        real_columns = [col for col in X.columns if any(var in col for var in Input_Variables)]
        other_columns = [col for col in X.columns if col not in real_columns]
        
        X_train_real = X_train[real_columns]
        X_train_other = X_train[other_columns]
        X_valid_real = X_valid[real_columns]
        X_valid_other = X_valid[other_columns]
        X_test_real = X_test[real_columns]
        X_test_other = X_test[other_columns]
        
        ss_train = MinMaxScaler()
        scaled_y_train = log10_1p(y_train.apply(pd.to_numeric, errors='coerce'))
        X_train_real.loc[:, out_var] = log10_1p(X_train_real[out_var])
        scaled_X_train_real = ss_train.fit_transform(X_train_real)
        scaled_X_train_real = pd.DataFrame(scaled_X_train_real, index = X_train_real.index, columns=X_train_real.columns)
        X_train_scaled = pd.concat([scaled_X_train_real, X_train_other], axis=1)
        cols = [out_var] + [col for col in X_train_scaled if col != out_var]
        X_train_scaled = X_train_scaled[cols]

        scaled_y_valid = log10_1p(y_valid.apply(pd.to_numeric, errors='coerce'))
        X_valid_real.loc[:, out_var] = log10_1p(X_valid_real[out_var])
        scaled_X_valid_real = ss_train.transform(X_valid_real)
        scaled_X_valid_real = pd.DataFrame(scaled_X_valid_real, index = X_valid_real.index, columns=X_valid_real.columns)
        X_valid_scaled = pd.concat([scaled_X_valid_real, X_valid_other], axis=1)
        cols = [out_var] + [col for col in X_valid_scaled if col != out_var]
        X_valid_scaled = X_valid_scaled[cols]

        scaled_y_test = log10_1p(y_test.apply(pd.to_numeric, errors='coerce'))
        X_test_real.loc[:, out_var] = log10_1p(X_test_real[out_var])
        scaled_X_test_real = ss_train.transform(X_test_real)
        scaled_X_test_real = pd.DataFrame(scaled_X_test_real, index = X_test_real.index, columns=X_test_real.columns)
        X_test_scaled = pd.concat([scaled_X_test_real, X_test_other], axis=1)
        cols = [out_var] + [col for col in X_test_scaled if col != out_var]
        X_test_scaled = X_test_scaled[cols]

        # Create sequences
        XX_train, yy_train, flags_train_seq = create_sequences(X_train_scaled, out_var, Input_Sequence, Output_Sequence, scaled_y_train, flags=flags_train)
        XX_valid, yy_valid, flags_valid_seq = create_sequences(X_valid_scaled, out_var, Input_Sequence, Output_Sequence, scaled_y_valid, flags=flags_valid)
        XX_test, yy_test, flags_test_seq = create_sequences(X_test_scaled, out_var, Input_Sequence, Output_Sequence, scaled_y_test, flags=flags_test)

        Sc_tr_x.append(XX_train)
        Sc_tr_y.append(yy_train)
        Sc_tr_flags.append(flags_train_seq)
        
        Sc_vl_x.append(XX_valid)
        Sc_vl_y.append(yy_valid)
        Sc_vl_flags.append(flags_valid_seq)

        Sc_te_x.append(XX_test)
        Sc_te_y.append(yy_test)
        Sc_te_flags.append(flags_test_seq)

        print(f"Train sequences shape for {location_name}:", XX_train.shape, yy_train.shape)
        print(f"Validation sequences shape for {location_name}:", XX_valid.shape, yy_valid.shape)
        print(f"Test sequences shape for {location_name}:", XX_test.shape, yy_test.shape)

# numpy array to tensor
Sc_tr_x = np.concatenate(Sc_tr_x, axis=0)
Sc_tr_y = np.concatenate(Sc_tr_y, axis=0)
Sc_tr_flags = np.concatenate(Sc_tr_flags, axis=0)

Sc_vl_x = np.concatenate(Sc_vl_x, axis=0)
Sc_vl_y = np.concatenate(Sc_vl_y, axis=0)
Sc_vl_flags = np.concatenate(Sc_vl_flags, axis=0)

Sc_te_x = np.concatenate(Sc_te_x, axis=0)
Sc_te_y = np.concatenate(Sc_te_y, axis=0)
Sc_te_flags = np.concatenate(Sc_te_flags, axis=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create datasets and dataloaders
train_dataset = TimeSeriesDataset(Sc_tr_x, Sc_tr_y, Sc_tr_flags, device)
valid_dataset = TimeSeriesDataset(Sc_vl_x, Sc_vl_y, Sc_vl_flags, device)
test_dataset = TimeSeriesDataset(Sc_te_x, Sc_te_y, Sc_te_flags, device)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Print shapes to verify
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(valid_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")   

# Select hyperparmater
Dropout_list = []
lr_list = []
Num_LSTM_Layers_list = []
Hidden_Layer_Size_list = []
Embedding_Dim_list = []
Num_Attention_list = []
EPOCHS = 200
pickle_dir = f'/PATH/TO/YOUR/PRETRAIN/PICKLE/FILE'
os.makedirs(pickle_dir, exist_ok=True)

for Dropout in Dropout_list:
    for learning_rate in lr_list:
        for Num_LSTM_Layers in Num_LSTM_Layers_list:
            for Hidden_Layer_Size in Hidden_Layer_Size_list:
                for Embedding_Dim in Embedding_Dim_list:
                    for Num_Attention_Heads in Num_Attention_list:
                        print("Dropout: ", Dropout)
                        print("learning_rate: ", learning_rate)
                        print("Num_LSTM_Layers: ", Num_LSTM_Layers)
                        print("Hidden_Layer_Size: ", Hidden_Layer_Size)
                        print("Embedding_Dim: ", Embedding_Dim)
                        print("Num_Attention_Heads: ", Num_Attention_Heads)
                     
                        # Model
                        model = TFT(
                            quantiles=quantiles, 
                            dropout=Dropout, 
                            device=device, 
                            hidden_layer_size=Hidden_Layer_Size, 
                            num_lstm_layers=Num_LSTM_Layers, 
                            embedding_dim=Embedding_Dim, 
                            encoder_steps=Input_Sequence, 
                            num_attention_heads=Num_Attention_Heads, 
                            col_to_idx=col_to_idx, 
                            static_covariates=static_covariates, 
                            time_dependent_categorical=time_dependent_categorical, 
                            time_dependent_continuous=time_dependent_continuous, 
                            category_counts=category_counts, 
                            known_time_dependent=known_time_dependent, 
                            observed_time_dependent=observed_time_dependent, 
                        )
                        model = model.to(device)
                        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                        criterion = QuantileLoss(quantiles=quantiles)
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=15)

                        model = train_model_with_flag_masking(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, EPOCHS)
                        final_lr = optimizer.param_groups[0]['lr']
                        model.eval()
                        model.to("cpu")

                        model_save_path = os.path.join(
                            pickle_dir, 
                            f"dr{Dropout}_lr{learning_rate}_nl{Num_LSTM_Layers}_hid{Hidden_Layer_Size}_emb{Embedding_Dim}_atten{Num_Attention_Heads}_final_lr{final_lr:.6f}.pt"
                        )
                        # Save model
                        torch.save(model, model_save_path)

                        # Count number of pickle files saved
                        saved_pickles = [f for f in os.listdir(pickle_dir) if f.endswith('.pt')]
                        print(f"Total number of pickle files saved in {pickle_dir}: {len(saved_pickles)}")    