#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
논문 재현 메타 스크립트 (대화형 실행 관리)
"Generalizable deep learning forecasting of harmful algal blooms using transfer learning across river systems"

Cursor/VSCode에서 이 파일을 열고 실행 버튼(F5)만 누르면 메뉴가 뜹니다.
"""

import os
import sys
import runpy

# 프로젝트 경로 설정
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

import torch
import numpy as np

# 시드 고정
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def check_environment():
    """환경 확인"""
    print("=" * 70)
    print("Environment Check")
    print("=" * 70)
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Project Directory: {PROJECT_DIR}")
    print("=" * 70)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_data_files():
    """데이터 파일 확인"""
    from config import RIVER_DATA_INFO, TFT_RIVER_DATA_INFO

    print("\nData Files Status:")
    print("-" * 50)

    # DATA 폴더 (Baseline용)
    total_data = 0
    for river, info in RIVER_DATA_INFO.items():
        count = sum(1 for path in info['locations'].values() if os.path.exists(path))
        total_data += count

    # TFT_DATA 폴더 (TFT용)
    total_tft = 0
    for river, info in TFT_RIVER_DATA_INFO.items():
        count = sum(1 for path in info['locations'].values() if os.path.exists(path))
        total_tft += count

    print(f"  DATA folder (Baseline): {total_data}/26 sites")
    print(f"  TFT_DATA folder (TFT):  {total_tft}/26 sites")
    print("-" * 50)
    return total_data == 26 and total_tft == 26

def check_outputs():
    """출력 폴더 상태 확인"""
    from config import OUTPUT_DIR

    status = {}

    # Baseline 결과
    baseline_dir = os.path.join(OUTPUT_DIR, "baseline")
    if os.path.exists(baseline_dir):
        files = [f for f in os.listdir(baseline_dir) if f.endswith('.pkl') or f.endswith('.pt')]
        status['baseline'] = len(files)
    else:
        status['baseline'] = 0

    # Pretrain 결과
    pretrain_dir = os.path.join(OUTPUT_DIR, "pretrain")
    if os.path.exists(pretrain_dir):
        files = [f for f in os.listdir(pretrain_dir) if f.endswith('.pkl') or f.endswith('.pt')]
        status['pretrain'] = len(files)
    else:
        status['pretrain'] = 0

    # Finetune 결과
    finetune_count = 0
    for scheme in ['S1', 'S2', 'S3', 'S4']:
        scheme_dir = os.path.join(OUTPUT_DIR, "finetune", scheme)
        if os.path.exists(scheme_dir):
            files = [f for f in os.listdir(scheme_dir) if f.endswith('.pkl') or f.endswith('.pt')]
            finetune_count += len(files)
    status['finetune'] = finetune_count

    return status

def run_step_baseline():
    """Step 1: Baseline 모델 학습"""
    print("\n" + "=" * 70)
    print("Step 1: Baseline Models Training")
    print("=" * 70)
    print("Models: LSTM, BiLSTM, GRU, CLA, Transformer")
    print("Sites: 26 monitoring sites")
    print("Output: outputs/baseline/")
    print("-" * 70)

    script_path = os.path.join(PROJECT_DIR, "BASELINE_MODELS.py")
    if os.path.exists(script_path):
        original_cwd = os.getcwd()
        os.chdir(PROJECT_DIR)
        original_argv = sys.argv
        sys.argv = [script_path, "--model", "all"]
        try:
            runpy.run_path(script_path, run_name="__main__")
            return True
        except Exception as e:
            print(f"[ERROR] {e}")
            return False
        finally:
            sys.argv = original_argv
            os.chdir(original_cwd)
    else:
        print(f"[ERROR] Script not found: {script_path}")
        return False

def run_step_pretrain():
    """Step 2: TFT 모델 사전학습"""
    print("\n" + "=" * 70)
    print("Step 2: TFT Pre-training (Source Domain)")
    print("=" * 70)
    print("Source Sites: 20 sites (6 sites excluded)")
    print("Epochs: 200, Early Stopping: 50")
    print("Output: outputs/pretrain/")
    print("-" * 70)

    script_path = os.path.join(PROJECT_DIR, "PRETRAIN_TFT.py")
    if os.path.exists(script_path):
        original_cwd = os.getcwd()
        os.chdir(PROJECT_DIR)
        original_argv = sys.argv
        sys.argv = [script_path]
        try:
            runpy.run_path(script_path, run_name="__main__")
            return True
        except Exception as e:
            print(f"[ERROR] {e}")
            return False
        finally:
            sys.argv = original_argv
            os.chdir(original_cwd)
    else:
        print(f"[ERROR] Script not found: {script_path}")
        return False

def run_step_finetune():
    """Step 3: Transfer Learning Fine-tuning"""
    print("\n" + "=" * 70)
    print("Step 3: Transfer Learning Fine-tuning")
    print("=" * 70)
    print("TL Schemes: S1, S2, S3, S4")
    print("Target Sites: 26 sites")
    print("Epochs: 50")
    print("Output: outputs/finetune/S1~S4/")
    print("-" * 70)

    script_path = os.path.join(PROJECT_DIR, "FINE_TUNE_S1TO4.py")
    if os.path.exists(script_path):
        original_cwd = os.getcwd()
        os.chdir(PROJECT_DIR)
        original_argv = sys.argv
        sys.argv = [script_path]
        try:
            runpy.run_path(script_path, run_name="__main__")
            return True
        except Exception as e:
            print(f"[ERROR] {e}")
            return False
        finally:
            sys.argv = original_argv
            os.chdir(original_cwd)
    else:
        print(f"[ERROR] Script not found: {script_path}")
        return False

def run_all_steps():
    """전체 실험 순차 실행"""
    print("\n" + "#" * 70)
    print("# FULL EXPERIMENT - AUTO RUN ALL STEPS")
    print("#" * 70)

    steps = [
        ("Step 1: Baseline", run_step_baseline),
        ("Step 2: Pretrain", run_step_pretrain),
        ("Step 3: Finetune", run_step_finetune),
    ]

    for i, (name, func) in enumerate(steps, 1):
        print(f"\n>>> Starting {name} ({i}/3)")
        success = func()
        if not success:
            print(f"\n[WARNING] {name} failed. Continue? (y/n): ", end="")
            if input().lower() != 'y':
                print("Experiment stopped.")
                return

    print("\n" + "=" * 70)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

def show_menu():
    """대화형 메뉴 표시"""
    output_status = check_outputs()

    print("\n")
    print("=" * 70)
    print("  HAB Forecasting - Transfer Learning Experiment Manager")
    print("=" * 70)
    print("""
  Paper: "Generalizable deep learning forecasting of harmful algal blooms
          using transfer learning across river systems"
    """)
    print("-" * 70)
    print("  Output Status:")
    print(f"    [1] Baseline  : {output_status['baseline']} files")
    print(f"    [2] Pretrain  : {output_status['pretrain']} files")
    print(f"    [3] Finetune  : {output_status['finetune']} files")
    print("-" * 70)
    print("""
  Select an option:

    [1] Run Step 1: Baseline Models (LSTM, BiLSTM, GRU, CLA, Transformer)
    [2] Run Step 2: TFT Pre-training (Source Domain, 20 sites)
    [3] Run Step 3: Fine-tuning (Transfer Learning S1-S4, 26 sites)

    [a] Run ALL Steps (1 -> 2 -> 3) automatically

    [i] Show paper info & expected results
    [s] Show detailed status
    [q] Quit
    """)
    print("=" * 70)

def show_paper_info():
    """논문 정보 표시"""
    print("""
================================================================================
                            PAPER INFORMATION
================================================================================

[Study Overview]
- 26 monitoring sites across 4 major rivers in South Korea
  - Nakdong (NAK): 12 sites
  - Han (HAN): 9 sites
  - Geum (GUM): 3 sites
  - Yeongsan (YOUNG): 2 sites

[Models]
- Baseline (5): LSTM, Bi-LSTM, GRU, CNN-LSTM-Attention (CLA), Transformer
- Transfer Learning: TFT (Temporal Fusion Transformer)

[TL Schemes]
- S1: Full Fine-tuning (all parameters updated)
- S2: Model Freezing (only output layer updated)
- S3: Full Fine-tuning + Output Layer Initialization
- S4: Model Freezing + Output Layer Initialization

[Data Configuration]
- Input: 10 variables, 2-week sequence
- Output: 1-week forecast (Cyanobacteria cell count)
- Training:   2012-2021 (10 years)
- Validation: 2022 (1 year)
- Test:       2023 (1 year)

[Expected Results]
- Pre-TL R2:  0.35 ~ 0.50
- Post-TL R2: 0.47 ~ 0.60
- Best site:  up to R2 = 0.89
- Improvement: 12.51% ~ 82.49%

================================================================================
""")

def show_detailed_status():
    """상세 상태 표시"""
    from config import RIVER_DATA_INFO, TFT_RIVER_DATA_INFO, OUTPUT_DIR

    print("\n" + "=" * 70)
    print("DETAILED STATUS")
    print("=" * 70)

    # 데이터 파일 상세
    print("\n[DATA folder - Baseline models]")
    for river, info in RIVER_DATA_INFO.items():
        existing = [loc for loc, path in info['locations'].items() if os.path.exists(path)]
        missing = [loc for loc, path in info['locations'].items() if not os.path.exists(path)]
        print(f"  {river}: {len(existing)}/{len(info['locations'])} sites")
        if missing:
            print(f"    Missing: {', '.join(missing)}")

    print("\n[TFT_DATA folder - TFT models]")
    for river, info in TFT_RIVER_DATA_INFO.items():
        existing = [loc for loc, path in info['locations'].items() if os.path.exists(path)]
        missing = [loc for loc, path in info['locations'].items() if not os.path.exists(path)]
        print(f"  {river}: {len(existing)}/{len(info['locations'])} sites")
        if missing:
            print(f"    Missing: {', '.join(missing)}")

    # 출력 폴더 상세
    print("\n[Output folders]")
    for subdir in ["baseline", "pretrain", "finetune/S1", "finetune/S2", "finetune/S3", "finetune/S4"]:
        full_path = os.path.join(OUTPUT_DIR, subdir)
        if os.path.exists(full_path):
            files = os.listdir(full_path)
            print(f"  {subdir}: {len(files)} files")
        else:
            print(f"  {subdir}: (not created)")

    print("=" * 70)

def interactive_menu():
    """대화형 메뉴 루프"""
    set_seed(42)
    check_environment()
    count_data_files()

    while True:
        show_menu()
        choice = input("  Enter your choice: ").strip().lower()

        if choice == '1':
            run_step_baseline()
            input("\n  Press Enter to continue...")
        elif choice == '2':
            run_step_pretrain()
            input("\n  Press Enter to continue...")
        elif choice == '3':
            run_step_finetune()
            input("\n  Press Enter to continue...")
        elif choice == 'a':
            run_all_steps()
            input("\n  Press Enter to continue...")
        elif choice == 'i':
            show_paper_info()
            input("\n  Press Enter to continue...")
        elif choice == 's':
            show_detailed_status()
            input("\n  Press Enter to continue...")
        elif choice == 'q':
            print("\n  Goodbye!")
            break
        else:
            print("\n  Invalid choice. Please try again.")

def main():
    """메인 함수 - 명령줄 인자 또는 대화형 모드"""
    # 명령줄 인자가 있으면 직접 실행
    if len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser(description='HAB Forecasting Experiment')
        parser.add_argument('--step', type=str,
                           choices=['all', 'baseline', 'pretrain', 'finetune'],
                           help='Step to execute')
        args = parser.parse_args()

        set_seed(42)
        check_environment()
        count_data_files()

        if args.step == 'all':
            run_all_steps()
        elif args.step == 'baseline':
            run_step_baseline()
        elif args.step == 'pretrain':
            run_step_pretrain()
        elif args.step == 'finetune':
            run_step_finetune()
    else:
        # 인자 없으면 대화형 메뉴
        interactive_menu()

if __name__ == "__main__":
    main()
