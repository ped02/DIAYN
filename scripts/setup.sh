#!/bin/bash


## Get path
CUR_PATH=$(realpath "${BASH_SOURCE[0]}")

if [ -z $CUR_PATH ]; then
    echo "Source path error"
    exit 1
fi

CUR_DIR=$(dirname "$CUR_PATH")

if [ -z $CUR_DIR ]; then
    echo "Directory ${CUR_DIR} not found"
    exit 1
fi

PROJECT_ROOT_DIR=$(dirname "$CUR_DIR")

if [ -z $PROJECT_ROOT_DIR ]; then
    echo "Parent directory ${PROJECT_ROOT_DIR} not found"
    exit 1
fi

## Config file
CONFIG_DIR_PATH="$PROJECT_ROOT_DIR/config"
CONFIG_FILE_PATH="$CONFIG_DIR_PATH/dev_config.yaml"

EXTRACT_SCRIPT_PATH="$CUR_DIR/extract_value_from_key.sh"

## Set python executable

# Check for python3
if [ -z "$PYTHON_EXEC" ]; then
    PYTHON_EXEC=$(which python3)
fi

# Check for python
if [ -z "$PYTHON_EXEC" ]; then
    PYTHON_EXEC=$(which python)
fi

if [ -z "$PYTHON_EXEC" ]; then
    echo "Python not found"
    exit 1
fi

echo "Using python '$PYTHON_EXEC'"

## Check python version

PYTHON_MIN_VERSION_MAJOR=3
PYTHON_MIN_VERSION_MINOR=10

PYTHON_DET_VERION=$("$PYTHON_EXEC" -c "import sys; print(sys.version_info[:2])")

PYTHON_DET_VERSION_MAJOR=$(echo "$PYTHON_DET_VERION" | awk -F '[(),]' '{print $2}')
PYTHON_DET_VERSION_MINOR=$(echo "$PYTHON_DET_VERION" | awk -F '[(),]' '{print $3}')

if [ $PYTHON_DET_VERSION_MAJOR -lt $PYTHON_MIN_VERSION_MAJOR ] || [ $PYTHON_DET_VERSION_MAJOR -eq $PYTHON_MIN_VERSION_MAJOR -a $PYTHON_DET_VERSION_MINOR -lt $PYTHON_MIN_VERSION_MINOR ]; then
    echo "Error: Detected Python version $PYTHON_DET_VERSION_MAJOR.$PYTHON_DET_VERSION_MINOR is not supported. Please install Python $PYTHON_MIN_VERSION_MAJOR.$PYTHON_MIN_VERSION_MINOR or later."
    exit 1
fi

## Setup venv
ENV_NAME_KEY="virtual_env_name"
ENV_NAME=$("$EXTRACT_SCRIPT_PATH" "$CONFIG_FILE_PATH" "$ENV_NAME_KEY")

if [ ! $? -eq 0 ]; then
    echo "ENV_NAME_KEY '$ENV_NAME_KEY' not found in config '$CONFIG_FILE_PATH'"
    exit 1
fi

ENV_PATH="$PROJECT_ROOT_DIR/$ENV_NAME"

if [ ! -d "$ENV_PATH" ]; then
    echo "Creating directory '$ENV_PATH'"
    mkdir "$ENV_PATH"
fi

if [ ! -d "$ENV_PATH/bin" ]; then
    echo "Creating virtual env"
    "$PYTHON_EXEC" -m venv "$ENV_PATH"
fi

source "$ENV_PATH/bin/activate"

ENV_PYTHON_EXEC="$ENV_PATH/bin/python"

## Install requirements.txt

REQUIREMENTS_PATH="$PROJECT_ROOT_DIR/requirements-dev.txt"

"$ENV_PYTHON_EXEC" -m pip install -r "$REQUIREMENTS_PATH"

## Check for conan c++
