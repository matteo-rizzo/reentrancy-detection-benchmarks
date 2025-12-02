#!/usr/bin/env bash

ROOT="$1"
[ -z "$ROOT" ] && { echo "Specifica la cartella root"; exit 1; }

find "$ROOT" -depth -type d | while read -r DIR; do
    [ "$DIR" = "$ROOT" ] && continue 

    if ! find "$DIR" -mindepth 1 -maxdepth 1 -type d | grep -q . ; then
        if ! find "$DIR" -type f -name "*.log" | grep -q . ; then
            echo "Removing folder $DIR"
            rm -rf "$DIR"
        fi
    fi
done



