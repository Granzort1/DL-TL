"""
논문 재현을 위한 설정 파일
"Generalizable deep learning forecasting of harmful algal blooms using transfer learning across river systems"

이 파일에서 경로와 하이퍼파라미터를 수정하세요.
"""

import os

# ============================================================
# 기본 경로 설정
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "DATA")
TFT_DATA_DIR = os.path.join(BASE_DIR, "TFT_DATA")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "pretrain"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "baseline"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "finetune"), exist_ok=True)

# ============================================================
# 데이터셋 설정 - 26개 모니터링 사이트
# ============================================================
RIVER_DATA_INFO = {
    'NAK': {  # 낙동강 (Nakdong River) - 12개 사이트
        'locations': {
            'SJB': os.path.join(DATA_DIR, 'NAK', 'SJB_nak_total.csv'),      # N1
            'NDB': os.path.join(DATA_DIR, 'NAK', 'NDB_nak_total.csv'),      # N2
            'GMB': os.path.join(DATA_DIR, 'NAK', 'GMB_nak_total.csv'),      # N3
            'DSB': os.path.join(DATA_DIR, 'NAK', 'DSB_nak_total.csv'),      # N4
            'HCB': os.path.join(DATA_DIR, 'NAK', 'HCB_nak_total.csv'),      # N5
            'HP': os.path.join(DATA_DIR, 'NAK', 'HP_nak_total.csv'),        # N6
            'GJGR': os.path.join(DATA_DIR, 'NAK', 'GJGR_nak_total.csv'),    # N7
            'GJGRB': os.path.join(DATA_DIR, 'NAK', 'GJGRB_nak_total.csv'),  # N8
            'CS': os.path.join(DATA_DIR, 'NAK', 'CS_nak_total.csv'),        # N9
            'MGMR': os.path.join(DATA_DIR, 'NAK', 'MGMR_nak_total.csv'),    # N10
            'CHB': os.path.join(DATA_DIR, 'NAK', 'CHB_nak_total.csv'),      # N11
            'CGB': os.path.join(DATA_DIR, 'NAK', 'CGB_nak_total.csv'),      # N12
        }
    },
    'HAN': {  # 한강 (Han River) - 9개 사이트
        'locations': {
            'YPB': os.path.join(DATA_DIR, 'HAN', 'YPB_han_total.csv'),      # H1
            'YJB': os.path.join(DATA_DIR, 'HAN', 'YJB_han_total.csv'),      # H2
            'GJG': os.path.join(DATA_DIR, 'HAN', 'GJG_han_total.csv'),      # H3
            'YC': os.path.join(DATA_DIR, 'HAN', 'YC_han_total.csv'),        # H4
            'GCB': os.path.join(DATA_DIR, 'HAN', 'GCB_han_total.csv'),      # H5
            'GDDG': os.path.join(DATA_DIR, 'HAN', 'GDDG_han_total.csv'),    # H6
            'HGDG': os.path.join(DATA_DIR, 'HAN', 'HGDG_han_total.csv'),    # H7
            'MSDG': os.path.join(DATA_DIR, 'HAN', 'MSDG_han_total.csv'),    # H8
            'JSCG': os.path.join(DATA_DIR, 'HAN', 'JSCG_han_total.csv'),    # H9
        }
    },
    'GUM': {  # 금강 (Geum River) - 3개 사이트
        'locations': {
            'GJB': os.path.join(DATA_DIR, 'GUM', 'GJB_gum_total.csv'),      # G1
            'BJB': os.path.join(DATA_DIR, 'GUM', 'BJB_gum_total.csv'),      # G2
            'SaJB': os.path.join(DATA_DIR, 'GUM', 'SaJB_gum_total.csv'),    # G3
        }
    },
    'YOUNG': {  # 영산강 (Yeongsan River) - 2개 사이트
        'locations': {
            'JSB': os.path.join(DATA_DIR, 'YOUNG', 'JSB_young_total.csv'),  # Y1
            'SCB': os.path.join(DATA_DIR, 'YOUNG', 'SCB_young_total.csv'),  # Y2
        }
    }
}

# TFT 모델용 데이터 (동일 구조, DATA -> TFT_DATA로 경로 변경)
def _convert_to_tft_path(path):
    """DATA 폴더 경로를 TFT_DATA로 변환 (Windows/Unix 호환)"""
    return path.replace('\\DATA\\', '\\TFT_DATA\\').replace('/DATA/', '/TFT_DATA/')

TFT_RIVER_DATA_INFO = {
    'NAK': {
        'locations': {k: _convert_to_tft_path(v) for k, v in RIVER_DATA_INFO['NAK']['locations'].items()}
    },
    'HAN': {
        'locations': {k: _convert_to_tft_path(v) for k, v in RIVER_DATA_INFO['HAN']['locations'].items()}
    },
    'GUM': {
        'locations': {k: _convert_to_tft_path(v) for k, v in RIVER_DATA_INFO['GUM']['locations'].items()}
    },
    'YOUNG': {
        'locations': {k: _convert_to_tft_path(v) for k, v in RIVER_DATA_INFO['YOUNG']['locations'].items()}
    }
}

# ============================================================
# 모델 하이퍼파라미터 (논문 Table S1 참조)
# ============================================================

# 공통 설정
INPUT_VARIABLES = ['Cyanocell', 'WT', 'Chla', 'TN', 'TP', 'WL', 'Discharge', 'Temp', 'Prec', 'Forecast']
INPUT_SEQUENCE = 2  # 2주 입력
OUTPUT_SEQUENCE = 1  # 1주 예측
BATCH_SIZE = 32
PRETRAIN_EPOCHS = 200
FINETUNE_EPOCHS = 50
PATIENCE = 50  # Early stopping

# TFT 하이퍼파라미터 (Grid Search 범위)
TFT_HYPERPARAMS = {
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [0.001, 0.0005, 0.0001],
    'num_lstm_layers': [1, 2],
    'hidden_layer_size': [32, 64, 128],
    'embedding_dim': [16, 32],
    'num_attention_heads': [1, 2, 4],
    'quantiles': [0.2, 0.5, 0.8]
}

# LSTM/Bi-LSTM/GRU 하이퍼파라미터
RNN_HYPERPARAMS = {
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [0.001, 0.0005, 0.0001],
    'num_layers': [1, 2],
    'hidden_dim': [32, 64, 128]
}

# Transformer 하이퍼파라미터
TRANSFORMER_HYPERPARAMS = {
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [0.001, 0.0005, 0.0001],
    'd_model': [32, 64],
    'nhead': [2, 4],
    'num_encoder_layers': [2, 4],
    'dim_feedforward': [128, 256]
}

# CNN-LSTM with Attention (CLA) 하이퍼파라미터
CLA_HYPERPARAMS = {
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [0.001, 0.0005, 0.0001],
    'num_layers': [1, 2],
    'hidden_dim': [32, 64, 128],
    'cnn_filters': [32, 64],
    'kernel_size': [3]
}

# ============================================================
# Transfer Learning 스킴 설정
# ============================================================
TL_SCHEMES = {
    'S1': 'Full Fine-tuning',           # 모든 파라미터 업데이트
    'S2': 'Model Freezing',             # 출력층만 학습
    'S3': 'Full Fine-tuning + Init',    # 출력층 초기화 후 전체 학습
    'S4': 'Model Freezing + Init'       # 출력층 초기화 후 출력층만 학습
}

# ============================================================
# 데이터 분할 기간
# ============================================================
DATA_SPLIT = {
    'train': ('2012-01-01', '2021-12-31'),
    'valid': ('2022-01-01', '2022-12-31'),
    'test': ('2023-01-01', '2023-12-31')
}

# ============================================================
# 소스 도메인 제외 사이트 (결측 데이터 과다)
# ============================================================
EXCLUDED_FROM_SOURCE = {
    'NAK': ['CGB', 'CHB', 'MGMR', 'CS'],  # 낙동강 4개 사이트 제외
    'HAN': ['YPB', 'YJB']                  # 한강 2개 사이트 제외
}

print(f"Configuration loaded. Base directory: {BASE_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
