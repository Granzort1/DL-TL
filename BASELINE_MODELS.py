"""
Baseline Models Training Script
- LSTM, Bi-LSTM, GRU, CNN-LSTM with Attention (CLA), Transformer
- 26개 모니터링 사이트 전체에 대해 학습
- 하이퍼파라미터 그리드 서치 수행

실행 방법:
    poetry run python BASELINE_MODELS.py --model all      # 모든 모델 학습
    poetry run python BASELINE_MODELS.py --model LSTM     # LSTM만 학습
    poetry run python BASELINE_MODELS.py --model BILSTM   # Bi-LSTM만 학습
    poetry run python BASELINE_MODELS.py --model GRU      # GRU만 학습
    poetry run python BASELINE_MODELS.py --model CLA      # CLA만 학습
    poetry run python BASELINE_MODELS.py --model TRANSFORMER  # Transformer만 학습
"""

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
import os
import glob
import argparse
import math
import pickle
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Set seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# Data paths - 26개 모니터링 사이트
# =============================================================================
river_data_info = {
    'NAK': {  # 낙동강 - 12개 사이트
        'locations': {
            'SJB': 'C:/DL-TL/DATA/NAK/SJB_nak_total.csv',
            'NDB': 'C:/DL-TL/DATA/NAK/NDB_nak_total.csv',
            'GMB': 'C:/DL-TL/DATA/NAK/GMB_nak_total.csv',
            'DSB': 'C:/DL-TL/DATA/NAK/DSB_nak_total.csv',
            'HCB': 'C:/DL-TL/DATA/NAK/HCB_nak_total.csv',
            'HP': 'C:/DL-TL/DATA/NAK/HP_nak_total.csv',
            'GJGR': 'C:/DL-TL/DATA/NAK/GJGR_nak_total.csv',
            'GJGRB': 'C:/DL-TL/DATA/NAK/GJGRB_nak_total.csv',
            'CS': 'C:/DL-TL/DATA/NAK/CS_nak_total.csv',
            'MGMR': 'C:/DL-TL/DATA/NAK/MGMR_nak_total.csv',
            'CHB': 'C:/DL-TL/DATA/NAK/CHB_nak_total.csv',
            'CGB': 'C:/DL-TL/DATA/NAK/CGB_nak_total.csv',
        }
    },
    'HAN': {  # 한강 - 9개 사이트
        'locations': {
            'YPB': 'C:/DL-TL/DATA/HAN/YPB_han_total.csv',
            'YJB': 'C:/DL-TL/DATA/HAN/YJB_han_total.csv',
            'GJG': 'C:/DL-TL/DATA/HAN/GJG_han_total.csv',
            'YC': 'C:/DL-TL/DATA/HAN/YC_han_total.csv',
            'GCB': 'C:/DL-TL/DATA/HAN/GCB_han_total.csv',
            'GDDG': 'C:/DL-TL/DATA/HAN/GDDG_han_total.csv',
            'HGDG': 'C:/DL-TL/DATA/HAN/HGDG_han_total.csv',
            'MSDG': 'C:/DL-TL/DATA/HAN/MSDG_han_total.csv',
            'JSCG': 'C:/DL-TL/DATA/HAN/JSCG_han_total.csv',
        }
    },
    'GUM': {  # 금강 - 3개 사이트
        'locations': {
            'GJB': 'C:/DL-TL/DATA/GUM/GJB_gum_total.csv',
            'BJB': 'C:/DL-TL/DATA/GUM/BJB_gum_total.csv',
            'SaJB': 'C:/DL-TL/DATA/GUM/SaJB_gum_total.csv',
        }
    },
    'YOUNG': {  # 영산강 - 2개 사이트
        'locations': {
            'JSB': 'C:/DL-TL/DATA/YOUNG/JSB_young_total.csv',
            'SCB': 'C:/DL-TL/DATA/YOUNG/SCB_young_total.csv',
        }
    }
}

# Global parameters
Input_Variables = ['Cyanocell', 'WT', 'Chla', 'TN', 'TP', 'WL', 'Discharge', 'Temp', 'Prec', 'Forecast']
batch_size = 32
Input_Sequence = 2
Output_Sequence = 1
EPOCHS = 200
PATIENCE = 50

# =============================================================================
# Model Definitions
# =============================================================================

class LSTMModel(nn.Module):
    """Standard LSTM Model"""
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
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM Model"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.0):
        super(BiLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class GRUModel(nn.Module):
    """GRU Model"""
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
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


class CLA(nn.Module):
    """CNN-LSTM with Attention Model"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, cnn_filters=64, kernel_size=3, dropout=0.0):
        super(CLA, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # CNN for feature extraction
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters, kernel_size=kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # LSTM for temporal sequence modeling
        self.lstm = nn.LSTM(cnn_filters, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # Attention mechanism
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)

        # Fully connected layer for output
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def attention(self, lstm_output):
        attention_scores = self.attention_weights(lstm_output)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        lstm_output, _ = self.lstm(x, (h0, c0))
        context_vector, attention_weights = self.attention(lstm_output)
        context_vector = self.dropout(context_vector)
        out = self.fc(context_vector)
        return out


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
    """Transformer Model for Time Series"""
    def __init__(self, input_dim, d_model=64, nhead=8, num_encoder_layers=4, dim_feedforward=256, dropout=0.1, output_dim=1):
        super(TransformerTimeSeries, self).__init__()

        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                         dim_feedforward=dim_feedforward,
                                                         dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.input_embedding(src)
        src = self.positional_encoding(src)
        src = src.permute(1, 0, 2)
        encoded_output = self.transformer_encoder(src)
        output = self.output_layer(encoded_output[-1])
        return output


# =============================================================================
# Dataset and Utility Functions
# =============================================================================

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y, flags, device):
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.float32).to(device)
        self.flags = torch.tensor(flags, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.flags[index]


def log10_1p(x):
    return np.log10(1 + x)


def filter_columns(df, variables):
    return df[[col for col in df.columns if any(var in col for var in variables)]]


def standardize_columns(df, location_name):
    """
    컬럼명을 표준화 (사이트 코드 제거)
    예: Cyanocell_SJB -> Cyanocell, WT_SJB -> WT
    중복 컬럼(WL_UP_ADD, Discharge_UP_ADD)은 제거
    """
    rename_map = {}
    drop_cols = []
    standard_vars = ['Cyanocell', 'WT', 'Chla', 'TN', 'TP', 'WL', 'Discharge', 'Temp', 'Prec']

    for col in df.columns:
        # 사이트 코드가 붙은 컬럼 처리 (우선 사용)
        for var in standard_vars:
            if col == f'{var}_{location_name}':
                rename_map[col] = var
                break

        # Forecast_Temp_XXX -> Forecast
        if col.startswith('Forecast_Temp_'):
            rename_map[col] = 'Forecast'

    # WL_UP_ADD, Discharge_UP_ADD는 사이트 특정 컬럼이 없을 때만 사용
    if 'WL' not in rename_map.values() and 'WL_UP_ADD' in df.columns:
        rename_map['WL_UP_ADD'] = 'WL'
    elif 'WL_UP_ADD' in df.columns:
        drop_cols.append('WL_UP_ADD')

    if 'Discharge' not in rename_map.values() and 'Discharge_UP_ADD' in df.columns:
        rename_map['Discharge_UP_ADD'] = 'Discharge'
    elif 'Discharge_UP_ADD' in df.columns:
        drop_cols.append('Discharge_UP_ADD')

    # 중복 컬럼 제거 후 이름 변경
    df = df.drop(columns=drop_cols, errors='ignore')
    return df.rename(columns=rename_map)


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
            interpolation_flags[column] = np.where(np.isnan(values), 1, 0)

    return df_copy, interpolation_flags


def create_sequences(data, target, seq_len, out_seq, target_values, flags=None):
    period = list(data.index)
    targets = list(data.columns)

    start_ = period.index(data.loc[~data[target].isna(), :].index[0])
    end_ = period.index(data.loc[~data[target].isna(), :].index[-1])

    d = np.array(data)[start_: end_ + 1, :]
    t = np.array(target_values)[start_: end_ + 1]
    f = np.array(flags)[start_: end_ + 1, :] if flags is not None else None

    xxs, yys, flags_list = [], [], []
    for i in range(len(d) - seq_len - out_seq + 1):
        xx = d[i:(i + seq_len + out_seq), :]
        yy = t[i + seq_len + out_seq - 1]
        flag = f[i + seq_len + out_seq - 1, targets.index(target)] if flags is not None else 0

        xxs.append(xx)
        yys.append(yy)
        if flags is not None:
            flags_list.append(flag)

    xxs_array = np.array(xxs)
    yys_array = np.array(yys).reshape(-1, 1)
    flags_array = np.array(flags_list).reshape(-1, 1) if flags is not None else None

    return xxs_array, yys_array, flags_array


def train_model_with_flag_masking(model, train_dataloader, valid_dataloader, criterion,
                                   optimizer, scheduler, epochs, patience=50):
    best_validation_loss = float('inf')
    no_improve = 0
    last_lr = optimizer.param_groups[0]['lr']
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        count_actual_values_train = 0

        for batch_idx, (train_inputs, train_targets, train_flags) in enumerate(train_dataloader):
            optimizer.zero_grad()
            train_outputs = model(train_inputs)

            # Flatten flags to match output shape [batch, 1]
            mask = (train_flags == 0).view(-1, 1)

            train_outputs_masked = train_outputs * mask.float()
            train_targets_masked = train_targets * mask.float()

            if mask.sum() > 0:
                train_loss = criterion(train_outputs_masked, train_targets_masked)
                train_loss.backward()
                optimizer.step()
                total_train_loss += train_loss.item()
                count_actual_values_train += 1

        avg_train_loss = total_train_loss / count_actual_values_train if count_actual_values_train > 0 else float('inf')

        model.eval()
        validation_loss = 0
        count_actual_values = 0
        with torch.no_grad():
            for batch_idx, (valid_inputs, valid_targets, valid_flags) in enumerate(valid_dataloader):
                valid_outputs = model(valid_inputs)

                # Flatten flags to match output shape [batch, 1]
                mask = (valid_flags == 0).view(-1, 1)

                valid_outputs_masked = valid_outputs * mask.float()
                valid_targets_masked = valid_targets * mask.float()

                if mask.sum() > 0:
                    valid_loss = criterion(valid_outputs_masked, valid_targets_masked)
                    validation_loss += valid_loss.item()
                    count_actual_values += 1

        if count_actual_values > 0:
            validation_loss /= count_actual_values

        current_lr = optimizer.param_groups[0]['lr']
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.4f}, Valid Loss: {validation_loss:.4f}, LR: {current_lr:.6f}")

        if current_lr != last_lr:
            print(f"  Learning rate changed to {current_lr:.6f} at epoch {epoch}")
            last_lr = current_lr

        scheduler.step(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f'  Early stopping at epoch {epoch}, best validation loss: {best_validation_loss:.4f}')
            if best_model_state:
                model.load_state_dict(best_model_state)
            return model

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model


# =============================================================================
# Hyperparameter configurations (논문 Table S1 참조)
# =============================================================================
HYPERPARAMS = {
    'LSTM': {
        'dropout': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'num_layers': [1, 2],
        'hidden_dim': [32, 64, 128]
    },
    'BILSTM': {
        'dropout': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'num_layers': [1, 2],
        'hidden_dim': [32, 64, 128]
    },
    'GRU': {
        'dropout': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'num_layers': [1, 2],
        'hidden_dim': [32, 64, 128]
    },
    'CLA': {
        'dropout': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'num_layers': [1, 2],
        'hidden_dim': [32, 64, 128],
        'cnn_filters': [32, 64],
        'kernel_size': [3]
    },
    'TRANSFORMER': {
        'dropout': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.0005, 0.0001],
        'd_model': [32, 64],
        'nhead': [2, 4],
        'num_encoder_layers': [2, 4],
        'dim_feedforward': [128, 256]
    }
}


def get_model(model_type, input_dim, output_dim, **kwargs):
    """모델 타입에 따라 적절한 모델 인스턴스 반환"""
    if model_type == 'LSTM':
        return LSTMModel(input_dim, kwargs['hidden_dim'], kwargs['num_layers'], output_dim, kwargs['dropout'])
    elif model_type == 'BILSTM':
        return BiLSTMModel(input_dim, kwargs['hidden_dim'], kwargs['num_layers'], output_dim, kwargs['dropout'])
    elif model_type == 'GRU':
        return GRUModel(input_dim, kwargs['hidden_dim'], kwargs['num_layers'], output_dim, kwargs['dropout'])
    elif model_type == 'CLA':
        return CLA(input_dim, kwargs['hidden_dim'], kwargs['num_layers'], output_dim,
                   kwargs['cnn_filters'], kwargs['kernel_size'], kwargs['dropout'])
    elif model_type == 'TRANSFORMER':
        return TransformerTimeSeries(input_dim, kwargs['d_model'], kwargs['nhead'],
                                      kwargs['num_encoder_layers'], kwargs['dim_feedforward'],
                                      kwargs['dropout'], output_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_baseline_model(model_type, river_name, location_name, filepath):
    """특정 사이트에 대해 baseline 모델 학습"""
    print(f"\n{'='*60}")
    print(f"Training {model_type} for {river_name}/{location_name}")
    print(f"{'='*60}")

    # 출력 디렉토리 설정
    save_dir = f'C:/DL-TL/outputs/baseline/{model_type}/{river_name}/{location_name}'
    os.makedirs(save_dir, exist_ok=True)

    # 데이터 로드 및 전처리
    try:
        df = pd.read_csv(filepath, encoding='cp949', index_col='Date')
    except:
        df = pd.read_csv(filepath, encoding='utf-8', index_col='Date')

    df.index = pd.to_datetime(df.index).normalize()

    # Kalman filter interpolation
    interpolated_data, interpolation_flags = kalman_filter_interpolation(df)
    temp_df = filter_columns(interpolated_data, Input_Variables)
    flags_df = filter_columns(interpolation_flags, Input_Variables)

    # 동적으로 target 컬럼 찾기 (원본 스타일)
    target_col = [col for col in temp_df.columns if 'Cyanocell' in col][0]
    temp_df_log = temp_df.copy()
    temp_df_log[target_col] = log10_1p(temp_df[target_col])

    # Normalization
    scaler = MinMaxScaler()
    temp_df_scaled = pd.DataFrame(scaler.fit_transform(temp_df_log),
                                   index=temp_df_log.index, columns=temp_df_log.columns)

    # Data split (2012-2021: train, 2022: valid, 2023: test)
    train_df = temp_df_scaled[temp_df_scaled.index.year <= 2021]
    valid_df = temp_df_scaled[temp_df_scaled.index.year == 2022]
    test_df = temp_df_scaled[temp_df_scaled.index.year == 2023]

    train_flags = flags_df[flags_df.index.year <= 2021]
    valid_flags = flags_df[flags_df.index.year == 2022]
    test_flags = flags_df[flags_df.index.year == 2023]

    # Create sequences
    XX_train, yy_train, flags_train = create_sequences(
        train_df, target_col, Input_Sequence, Output_Sequence,
        temp_df_scaled[target_col], flags_df
    )
    XX_valid, yy_valid, flags_valid = create_sequences(
        valid_df, target_col, Input_Sequence, Output_Sequence,
        temp_df_scaled[target_col], flags_df
    )
    XX_test, yy_test, flags_test = create_sequences(
        test_df, target_col, Input_Sequence, Output_Sequence,
        temp_df_scaled[target_col], flags_df
    )

    if len(XX_train) == 0 or len(XX_valid) == 0:
        print(f"  Skipping {location_name}: insufficient data")
        return None

    print(f"  Train: {XX_train.shape}, Valid: {XX_valid.shape}, Test: {XX_test.shape}")

    # Create dataloaders
    train_dataset = TimeSeriesDataset(XX_train, yy_train, flags_train, device)
    valid_dataset = TimeSeriesDataset(XX_valid, yy_valid, flags_valid, device)
    test_dataset = TimeSeriesDataset(XX_test, yy_test, flags_test, device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = XX_train.shape[2]
    output_dim = 1

    # Hyperparameter grid search
    params = HYPERPARAMS[model_type]
    best_valid_loss = float('inf')
    best_model = None
    best_params = None

    # Generate all hyperparameter combinations
    if model_type in ['LSTM', 'BILSTM', 'GRU']:
        for dropout in params['dropout']:
            for lr in params['learning_rate']:
                for num_layers in params['num_layers']:
                    for hidden_dim in params['hidden_dim']:
                        model = get_model(model_type, input_dim, output_dim,
                                         dropout=dropout, num_layers=num_layers,
                                         hidden_dim=hidden_dim).to(device)

                        criterion = nn.MSELoss()
                        optimizer = optim.Adam(model.parameters(), lr=lr)
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                          factor=0.5, patience=10)

                        print(f"  Training: dr={dropout}, lr={lr}, layers={num_layers}, hidden={hidden_dim}")
                        model = train_model_with_flag_masking(model, train_loader, valid_loader,
                                                              criterion, optimizer, scheduler,
                                                              EPOCHS, PATIENCE)

                        # Evaluate on validation
                        model.eval()
                        valid_loss = 0
                        with torch.no_grad():
                            for inputs, targets, flags in valid_loader:
                                outputs = model(inputs)
                                mask = flags == 0
                                if mask.sum() > 0:
                                    valid_loss += criterion(outputs[mask], targets[mask]).item()

                        if valid_loss < best_valid_loss:
                            best_valid_loss = valid_loss
                            best_model = model
                            best_params = {'dropout': dropout, 'lr': lr,
                                          'num_layers': num_layers, 'hidden_dim': hidden_dim}

    elif model_type == 'CLA':
        for dropout in params['dropout']:
            for lr in params['learning_rate']:
                for num_layers in params['num_layers']:
                    for hidden_dim in params['hidden_dim']:
                        for cnn_filters in params['cnn_filters']:
                            for kernel_size in params['kernel_size']:
                                model = get_model(model_type, input_dim, output_dim,
                                                 dropout=dropout, num_layers=num_layers,
                                                 hidden_dim=hidden_dim, cnn_filters=cnn_filters,
                                                 kernel_size=kernel_size).to(device)

                                criterion = nn.MSELoss()
                                optimizer = optim.Adam(model.parameters(), lr=lr)
                                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                                  factor=0.5, patience=10)

                                print(f"  Training: dr={dropout}, lr={lr}, layers={num_layers}, hidden={hidden_dim}, filters={cnn_filters}")
                                model = train_model_with_flag_masking(model, train_loader, valid_loader,
                                                                      criterion, optimizer, scheduler,
                                                                      EPOCHS, PATIENCE)

                                model.eval()
                                valid_loss = 0
                                with torch.no_grad():
                                    for inputs, targets, flags in valid_loader:
                                        outputs = model(inputs)
                                        mask = flags == 0
                                        if mask.sum() > 0:
                                            valid_loss += criterion(outputs[mask], targets[mask]).item()

                                if valid_loss < best_valid_loss:
                                    best_valid_loss = valid_loss
                                    best_model = model
                                    best_params = {'dropout': dropout, 'lr': lr,
                                                  'num_layers': num_layers, 'hidden_dim': hidden_dim,
                                                  'cnn_filters': cnn_filters, 'kernel_size': kernel_size}

    elif model_type == 'TRANSFORMER':
        for dropout in params['dropout']:
            for lr in params['learning_rate']:
                for d_model in params['d_model']:
                    for nhead in params['nhead']:
                        for num_encoder_layers in params['num_encoder_layers']:
                            for dim_feedforward in params['dim_feedforward']:
                                model = get_model(model_type, input_dim, output_dim,
                                                 dropout=dropout, d_model=d_model, nhead=nhead,
                                                 num_encoder_layers=num_encoder_layers,
                                                 dim_feedforward=dim_feedforward).to(device)

                                criterion = nn.MSELoss()
                                optimizer = optim.Adam(model.parameters(), lr=lr)
                                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                                  factor=0.5, patience=10)

                                print(f"  Training: dr={dropout}, lr={lr}, d_model={d_model}, nhead={nhead}")
                                model = train_model_with_flag_masking(model, train_loader, valid_loader,
                                                                      criterion, optimizer, scheduler,
                                                                      EPOCHS, PATIENCE)

                                model.eval()
                                valid_loss = 0
                                with torch.no_grad():
                                    for inputs, targets, flags in valid_loader:
                                        outputs = model(inputs)
                                        mask = flags == 0
                                        if mask.sum() > 0:
                                            valid_loss += criterion(outputs[mask], targets[mask]).item()

                                if valid_loss < best_valid_loss:
                                    best_valid_loss = valid_loss
                                    best_model = model
                                    best_params = {'dropout': dropout, 'lr': lr,
                                                  'd_model': d_model, 'nhead': nhead,
                                                  'num_encoder_layers': num_encoder_layers,
                                                  'dim_feedforward': dim_feedforward}

    if best_model is None:
        print(f"  No valid model found for {location_name}")
        return None

    # Save best model
    model_path = f'{save_dir}/best_model.pt'
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'params': best_params,
        'scaler': scaler
    }, model_path)

    # Evaluate on test set
    best_model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets, flags in test_loader:
            outputs = best_model(inputs)
            mask = flags == 0
            if mask.sum() > 0:
                all_preds.extend(outputs[mask].cpu().numpy().flatten())
                all_targets.extend(targets[mask].cpu().numpy().flatten())

    if len(all_preds) > 0:
        r2 = r2_score(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        mae = mean_absolute_error(all_targets, all_preds)

        print(f"\n  Best params: {best_params}")
        print(f"  Test Results - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        # Save results
        results = {
            'model_type': model_type,
            'river': river_name,
            'location': location_name,
            'best_params': best_params,
            'r2': r2,
            'rmse': rmse,
            'mae': mae
        }
        with open(f'{save_dir}/results.pkl', 'wb') as f:
            pickle.dump(results, f)

        return results

    return None


def main():
    parser = argparse.ArgumentParser(description='Baseline Models Training')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'LSTM', 'BILSTM', 'GRU', 'CLA', 'TRANSFORMER'],
                       help='Model type to train')
    args = parser.parse_args()

    print("=" * 60)
    print("Baseline Models Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    if args.model == 'all':
        model_types = ['LSTM', 'BILSTM', 'GRU', 'CLA', 'TRANSFORMER']
    else:
        model_types = [args.model]

    all_results = []

    for model_type in model_types:
        print(f"\n{'#'*60}")
        print(f"Training {model_type} Model")
        print(f"{'#'*60}")

        for river_name, river_info in river_data_info.items():
            for location_name, filepath in river_info['locations'].items():
                if os.path.exists(filepath):
                    result = train_baseline_model(model_type, river_name, location_name, filepath)
                    if result:
                        all_results.append(result)
                else:
                    print(f"File not found: {filepath}")

    # Save summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv('C:/DL-TL/outputs/baseline/summary.csv', index=False)
        print("\n" + "=" * 60)
        print("Training Complete! Summary saved to outputs/baseline/summary.csv")
        print("=" * 60)
        print(summary_df.groupby('model_type')[['r2', 'rmse', 'mae']].mean())


if __name__ == "__main__":
    main()
