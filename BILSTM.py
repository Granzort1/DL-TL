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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
    
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Apply dropout to the last time step's output
        out = self.fc(out)
        return out

##### GRU
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.0):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])  # Apply dropout to the last time step's output
        out = self.fc(out)
        return out

##### Bi-LSTM
class BiLSTM(nn.Module):
    torch.manual_seed(42)
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.0):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Apply dropout to the last time step's output
        out = self.fc(out)
        return out
    
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

def filter_columns(df, variables):
    return df[[col for col in df.columns if any(var in col for var in variables)]]

def classify_stage(value):
    if value < 3:
        return 0
    elif value < 4:
        return 1
    elif value < 5:
        return 2
    elif value < 6:
        return 3
    else:
        return 4
    
def round_columns(df, keyword, decimals):
    for col in df.columns:
        if keyword in col.split('_')[0]:  # Check up to the first underscore
            df[col] = np.round(df[col].astype(float), decimals)
    return df
