#!/bin/bash

DEST_USER="root"
DEST_DIR="project/"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$1" == "shell" ]; then
    echo "Opening SSH shell to $DEST_USER@$DEST_HOST on port $DEST_PORT..."
    ssh -p "$DEST_PORT" "$DEST_USER@$DEST_HOST"
    exit 0
fi

if [ "$1" == "init" ]; then
    echo "Running init on $DEST_USER@$DEST_HOST on port $DEST_PORT..."
    cat $SCRIPT_DIR/init_commands | ssh -p "$DEST_PORT" "$DEST_USER@$DEST_HOST" "bash -"
    exit 0
fi

EXCLUDE_FILE="$SCRIPT_DIR/upload_exclude"
INCLUDE_FILE="$SCRIPT_DIR/upload_include"

echo "Excluding files from $EXCLUDE_FILE"

rsync \
        -urlavz \
        -e "ssh -p $DEST_PORT" \
        --exclude-from="$EXCLUDE_FILE" \
        --include-from="$INCLUDE_FILE" \
        . \
        "$DEST_USER@$DEST_HOST:$DEST_DIR"
