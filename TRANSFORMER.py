import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd
import numpy as np
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
pd.set_option('display.max_columns', None)
#import seaborn as sns
import os
import plotly.graph_objects as go
import glob
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from datetime import timedelta

# Set the font properties
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
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

import torch
import torch.nn as nn
import numpy as np
import math

# Positional Encoding for Sequential Data
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=8, num_encoder_layers=4, dim_feedforward=256, dropout=0.1, output_dim=1):
        super(TransformerTimeSeries, self).__init__()

        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src):
        """
        src: (batch_size, seq_len, input_dim) - Input time-series data
        """
        src = self.input_embedding(src)
        src = self.positional_encoding(src)

        src = src.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        encoded_output = self.transformer_encoder(src)

        output = self.output_layer(encoded_output[-1])  # Take last time-step representation
        return output
    
def log10_1p(x):
    return np.log10(1 + x) 

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, flags, device):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)
        self.flags = torch.tensor(flags, dtype=torch.float32).to(device)  # Add flags
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.flags[index]  # Return flags as well

def create_sequences_with_flags(data, flags, seq_len, fct_h, target_idx):
    xs, ys, fs = [], [], []
    for i in range(len(data) - seq_len - fct_h + 1):
        x = data[i:(i + seq_len), :]
        y = data[i + seq_len + fct_h - 1, target_idx]  # Use the specified target column
        f = flags[i + seq_len + fct_h - 1, target_idx]  # Use flag for the corresponding target column
        xs.append(x)
        ys.append(y)
        fs.append(f)
    return np.array(xs), np.array(ys).reshape(-1, 1), np.array(fs).reshape(-1, 1)

from pykalman import KalmanFilter

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
            train_outputs_masked = torch.where(mask, train_outputs, torch.tensor(0.0, device=train_outputs.device))
            train_targets_masked = torch.where(mask, train_targets, torch.tensor(0.0, device=train_targets.device))

            # If valid, calculate the training loss
            if mask.sum() > 0:
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
                # mask_expanded = mask.unsqueeze(-1).expand_as(valid_outputs)

                valid_outputs_masked = torch.where(mask, valid_outputs, torch.tensor(0.0, device=valid_outputs.device))
                valid_targets_masked = torch.where(mask, valid_targets, torch.tensor(0.0, device=valid_targets.device))

                # If valid, calculate the validation loss
                if mask.sum() > 0:
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
