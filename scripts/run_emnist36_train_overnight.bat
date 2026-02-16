@echo off
REM 過夜訓練：關掉 Cursor 也能跑。請先讓電腦「不要休眠」。
cd /d "%~dp0.."
echo [%date% %time%] Start EMNIST36 CNN 10 epochs, batch_size=10, CPU >> outputs\emnist36_overnight_log.txt
python scripts/train_emnist_digits_letters.py --model cnn --epochs 10 --save_best --batch_size 10 --cpu >> outputs\emnist36_overnight_log.txt 2>&1
echo [%date% %time%] Exit code: %ERRORLEVEL% >> outputs\emnist36_overnight_log.txt
pause
