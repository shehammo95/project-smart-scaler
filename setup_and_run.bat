@echo off
echo Installing required packages...
pip install pandas numpy
echo.
echo Running Smart Food Scale System...
python src/test_scale.py
pause 