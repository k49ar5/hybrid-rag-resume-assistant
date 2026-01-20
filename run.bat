@echo off
cls
color 0A

echo ========================================================
echo     HYBRID RAG SYSTEM LAUNCHER
echo ========================================================
echo.


echo [1/3] Starting Qdrant Database (Docker)...
docker-compose up -d
if %errorlevel% neq 0 (
    color 0C
    echo.
    echo [ERROR] Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b
)


echo.
echo [INFO] Waiting 3 seconds for database initialization...
timeout /t 3 /nobreak >nul


echo.
echo [2/3] Checking for .env file...
if not exist .env (
    color 0C
    echo.
    echo [ERROR] File .env is missing!
    echo Please create it and add your Ngrok URL.
    pause
    exit /b
)


echo.
echo [3/3] Starting FastAPI Server...
echo --------------------------------------------------------
echo    Swagger UI: http://localhost:8000/docs
echo --------------------------------------------------------
echo.

python api.py

pause