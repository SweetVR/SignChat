@echo off
setlocal
cd /d "%~dp0"

rem Pick a Python: the "py" launcher if present, otherwise "python".
set "PY=python"
where py >nul 2>nul && set "PY=py -3"

if not exist ".venv\Scripts\python.exe" (
    echo Setting up SignChat for the first time. This takes a minute or two...
    %PY% -m venv .venv || goto nopython
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto failed
)

start "" ".venv\Scripts\pythonw.exe" -m signchat
exit /b 0

:nopython
echo.
echo SignChat needs Python 3.10 or newer: https://www.python.org/downloads/
echo During installation, tick "Add python.exe to PATH", then run this file again.
pause
exit /b 1

:failed
echo.
echo Installing SignChat's requirements failed. See the messages above.
echo Delete the .venv folder and try again, or ask for help with the error text.
pause
exit /b 1
