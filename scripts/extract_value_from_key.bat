@echo off
setlocal enabledelayedexpansion

:: Check if the number of arguments is correct
if "%~2"=="" (
    echo Usage: %0 YAML_FILE KEY
    exit /b 1
)

:: Search for the key in the YAML file
for /f "tokens=2 delims=:" %%a in ('findstr /c:"%~2" "%~1"') do (
    set "value=%%a"

    if defined value (
        rem Remove leading spaces
        set "value=!value: =!"

        rem Check if the value is not empty
        if not "!value!"=="" (
            goto :break

        )
    )
)
:break

if defined value (
    echo !value!
    exit /b 0
) else (
    exit /b 1
)
