@echo off
echo Creating backup of Smart Food Scale System...

:: Create backup directory with timestamp
set TIMESTAMP=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set BACKUP_DIR=backup_%TIMESTAMP%

:: Create backup directory
mkdir %BACKUP_DIR%
mkdir %BACKUP_DIR%\data

:: Copy Python files
copy *.py %BACKUP_DIR%\
copy *.bat %BACKUP_DIR%\
copy *.txt %BACKUP_DIR%\
copy *.md %BACKUP_DIR%\

:: Copy data files
copy data\*.csv %BACKUP_DIR%\data\
copy data\*.json %BACKUP_DIR%\data\

:: Create backup info file
echo Backup created on %date% at %time% > %BACKUP_DIR%\backup_info.txt
echo. >> %BACKUP_DIR%\backup_info.txt
echo Files backed up: >> %BACKUP_DIR%\backup_info.txt
dir /b %BACKUP_DIR% >> %BACKUP_DIR%\backup_info.txt
echo. >> %BACKUP_DIR%\backup_info.txt
echo Data files backed up: >> %BACKUP_DIR%\backup_info.txt
dir /b %BACKUP_DIR%\data >> %BACKUP_DIR%\backup_info.txt

echo.
echo Backup completed successfully!
echo Backup location: %BACKUP_DIR%
echo.
pause 