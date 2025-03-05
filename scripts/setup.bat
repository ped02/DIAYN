@echo off

REM Get path

set "CUR_DIR=%~dp0"

if not exist "%CUR_DIR%" (
    echo Directory %CUR_DIR% not found
    exit /b 1
)

set "PROJECT_ROOT_DIR=%~dp0.."

if not exist "%PROJECT_ROOT_DIR%" (
    echo Parent directory %PROJECT_ROOT_DIR% not found
    exit /b 1
)

REM Set python executable

REM Check for python3
if not defined PYTHON_EXEC (
    for /f "delims=" %%i in ('where python3') do (
        set "PYTHON_EXEC=%%i"
        goto :break1
    )
)
:break1

REM Check for python
if not defined PYTHON_EXEC (
    for /f "delims=" %%i in ('where python') do (
        set "PYTHON_EXEC=%%i"
        goto :break2
    )
)
:break2

if not defined PYTHON_EXEC (
    echo Python not found
    exit /b 1
)

echo Using python '%PYTHON_EXEC%'

REM Config File
set "CONFIG_DIR_PATH=%PROJECT_ROOT_DIR%\config"
set "CONFIG_FILE_PATH=%CONFIG_DIR_PATH%\dev_config.yaml"
set "EXTRACT_SCRIPT_PATH=%CUR_DIR%\extract_value_from_key.bat"

REM Check python version

set "PYTHON_MIN_VERSION_MAJOR=3"
set "PYTHON_MIN_VERSION_MINOR=8"

for /f "delims=" %%i in ('%PYTHON_EXEC% -c "import sys; print(sys.version_info[:2])"') do set "PYTHON_DET_VERSION=%%i"

for /f "tokens=1,2 delims=()," %%a in ("%PYTHON_DET_VERSION%") do (
    set "PYTHON_DET_VERSION_MAJOR=%%a"
    set "PYTHON_DET_VERSION_MINOR=%%b"
)

if %PYTHON_DET_VERSION_MAJOR% lss %PYTHON_MIN_VERSION_MAJOR% (
    echo Error: Detected Python version %PYTHON_DET_VERSION_MAJOR%.%PYTHON_DET_VERSION_MINOR% is not supported. Please install Python %PYTHON_MIN_VERSION_MAJOR%.%PYTHON_MIN_VERSION_MINOR% or later.
    exit /b 1
) else if %PYTHON_DET_VERSION_MAJOR% equ %PYTHON_MIN_VERSION_MAJOR% if %PYTHON_DET_VERSION_MINOR% lss %PYTHON_MIN_VERSION_MINOR% (
    echo Error: Detected Python version %PYTHON_DET_VERSION_MAJOR%.%PYTHON_DET_VERSION_MINOR% is not supported. Please install Python %PYTHON_MIN_VERSION_MAJOR%.%PYTHON_MIN_VERSION_MINOR% or later.
    exit /b 1
)

REM Setup venv
set "ENV_NAME_KEY=virtual_env_name"
for /f "tokens=* delims=" %%a in ('%EXTRACT_SCRIPT_PATH% %CONFIG_FILE_PATH% %ENV_NAME_KEY%') do set "ENV_NAME=%%a"

if not %errorlevel% equ 0 (
    echo ENV_NAME_KEY '%ENV_NAME_KEY%' not found in config '%CONFIG_FILE_PATH%'
    exit /b 1
)

if not defined ENV_NAME (
    echo Error: Unable to extract environment name >&2
    exit /b 1
)

set "ENV_PATH=%PROJECT_ROOT_DIR%\%ENV_NAME%"

echo %ENV_PATH%

if not exist "%ENV_PATH%" (
    echo Creating directory '%ENV_PATH%'
    mkdir "%ENV_PATH%"
)

if not exist "%ENV_PATH%\Scripts" (
    echo Creating virtual env
    "%PYTHON_EXEC%" -m venv "%ENV_PATH%"
)

call "%ENV_PATH%\Scripts\activate"

set "ENV_PYTHON_EXEC=%ENV_PATH%\Scripts\python.exe"

REM Install requirements.txt

set "REQUIREMENTS_PATH=%PROJECT_ROOT_DIR%\requirements-dev.txt"

%ENV_PYTHON_EXEC% -m pip install -r "%REQUIREMENTS_PATH%"

REM Installing pre-commit
pre-commit install

REM Check for conan C++

REM (Add your conan C++ check here)

echo Script completed successfully
exit /b 0
