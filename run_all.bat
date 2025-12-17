@echo off
chcp 65001 > nul
echo ============================================================
echo 논문 재현 실험 실행
echo ============================================================
echo.
echo Poetry 환경: C:\Users\skadl\poetry\dl-tl
echo 프로젝트 폴더: C:\DL-TL
echo.

cd /d "C:\Users\skadl\poetry\dl-tl"

if "%1"=="" (
    echo 사용법: run_all.bat [옵션]
    echo.
    echo 옵션:
    echo   info      - 실행 정보 출력
    echo   baseline  - Baseline 모델 학습
    echo   pretrain  - TFT 사전학습
    echo   finetune  - Transfer Learning Fine-tuning
    echo   all       - 전체 실험 실행
    echo.
    echo 예시:
    echo   run_all.bat info
    echo   run_all.bat baseline
    echo   run_all.bat all
    echo.
    poetry run python "C:\DL-TL\run_experiment.py" --step info
) else (
    poetry run python "C:\DL-TL\run_experiment.py" --step %1
)
