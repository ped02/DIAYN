#!/bin/bash
MATCH=$(grep -w $2 $1)

if [ $? -eq 0 ]; then
    value=$(echo "$MATCH" | sed 's/^[^:]*: *//')
    # Check if the value is not empty
    if [ -n "$value" ]; then
        echo "$value"
        exit 0
    else
        # Exit with an error code if the value is empty
        echo "Error: Empty value for key '$2' in '$1'" >&2
        exit 1
    fi
else
    echo "Error: Key '$2' not found in '$1'" >&2
    exit 1
fi
