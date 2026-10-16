@echo off
setlocal
cd /d "%~dp0"
title FIAP CP2/CP3 - Big Data Architecture Copilot

echo ======================================================
echo  FIAP - Big Data Architecture Copilot - Demonstracao
echo ======================================================
echo.
echo Este modo exibe uma DEMONSTRACAO ILUSTRATIVA POR REGRAS.
echo Ele NAO executa fine-tuning, GPU ou uma LLM treinada.
echo.
where py >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON=py -3"
) else (
    set "PYTHON=python"
)
%PYTHON% --version
if errorlevel 1 (
    echo.
    echo Python 3 nao encontrado. Instale Python 3 e tente novamente.
    pause
    exit /b 1
)
%PYTHON% -m pip install -r requirements-ui.txt
if errorlevel 1 (
    echo.
    echo Falha ao instalar Streamlit. Verifique a conexao e o Python.
    pause
    exit /b 1
)
%PYTHON% -m streamlit run app/app.py
if errorlevel 1 (
    echo.
    echo O Streamlit nao iniciou. Consulte README.md.
    pause
    exit /b 1
)
endlocal
