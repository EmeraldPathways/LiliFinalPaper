@echo off
setlocal
cd /d "%~dp0frontend"
set "NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8009"
npm.cmd run build
if errorlevel 1 exit /b %errorlevel%
npm.cmd run start -- --port 3009
