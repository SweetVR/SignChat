@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Run "Start SignChat.bat" once first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements-webcam.txt
echo.
echo Done. In SignChat, open Settings and choose "Webcam" as the input.
pause
