@echo off
rem Builds a stand-alone SignChat.exe (no Python needed on the computers you share it with).
cd /d "%~dp0"
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
%PY% -m pip install -r requirements.txt pyinstaller || goto failed
%PY% -m PyInstaller --noconfirm signchat.spec || goto failed
echo.
echo Done: dist\SignChat\SignChat.exe   (zip the whole dist\SignChat folder to share it)
pause
exit /b 0
:failed
echo Build failed, see above.
pause
exit /b 1
