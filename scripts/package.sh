#!/bin/bash

## Input Validation
usage() {
    echo "Usage: $0 <path to output package> [-o]"
    echo "-o: optional flag for overwrite"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

PACKAGE_ROOT_PATH=$1
OVERWRITE=0

if [ $# -eq 2 ]; then
    if [ "$2" = "-o" ]; then
        OVERWRITE=1
    else
        usage
    fi
elif [ $# -gt 2 ]; then
    usage
fi

if [ -d "$PACKAGE_ROOT_PATH" ]; then
    if [ $OVERWRITE -eq 1 ]; then
        echo "Overwriting $PACKAGE_ROOT_PATH"
        rm -r "$PACKAGE_ROOT_PATH"
    else
        echo "File $PACKAGE_ROOT_PATH exists. Use -o to overwrite."
        exit 1
    fi
fi

## Creating package
echo "Creating package: $PACKAGE_ROOT_PATH"
mkdir -p "$PACKAGE_ROOT_PATH"

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

## Populating Package

rsync -avuP --exclude-from="${PROJECT_ROOT_DIR}/.rsync_exclude_package"  --include-from="${PROJECT_ROOT_DIR}/.rsync_include_package" --exclude="*" --prune-empty-dirs "${PROJECT_ROOT_DIR}/" "${PACKAGE_ROOT_PATH}"
