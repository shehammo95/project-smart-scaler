@echo off
echo Installing required packages...
pip install pandas numpy scikit-learn matplotlib joblib
echo.
echo Running ML-Enhanced Smart Food Scale System...
python test_scaler0.2.py
pause 