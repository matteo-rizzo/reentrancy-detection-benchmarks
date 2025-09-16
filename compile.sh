#!/usr/bin/env bash

BASE_DIR="${1:-.}"

rm -rf bins

find "$BASE_DIR" -type f -name "*.sol" | while read -r file; do
    echo "Compiling: $file"

    # Extract first pragma solidity line
    pragma_line=$(grep -m 1 "pragma solidity" "$file")
    if [[ -z "$pragma_line" ]]; then
        echo "  ⚠️ No pragma found, skipping..."
        continue
    fi

    # Extract a single clean version number
    version=$(echo "$pragma_line" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1)
    if [[ -z "$version" ]]; then
        echo "  ⚠️ Could not parse version from pragma: $pragma_line"
        continue
    fi

    echo "  → Using solc $version"

    # Switch compiler version (auto-install if missing)
    if ! solc-select use "$version" >/dev/null 2>&1; then
        echo "  ⚠️ solc version $version not installed. Installing..."
        if solc-select install "$version"; then
            solc-select use "$version"
        else
            echo "  ❌ Failed to install solc $version"
            continue
        fi
    fi

    # Determine folder type
    folder_type="UNKNOWN"
    if [[ "$file" == *"/safe/"* ]]; then
        folder_type="SAFE"
    elif [[ "$file" == *"/reentrant/"* ]]; then
        folder_type="REENTRANT"
    fi

    # Use temporary folder first
    tmp_outdir=$(mktemp -d)

    if ! solc --bin --optimize --overwrite -o "$tmp_outdir" "$file"; then
        echo "  ❌ Compilation failed for $file"
        rm -rf "$tmp_outdir"
        continue
    fi

    # Check if any non-empty .bin exists
    has_output=false
    for binfile in "$tmp_outdir"/*.bin; do
        [[ -s "$binfile" ]] && has_output=true && break
    done

    if [ "$has_output" = true ]; then
        # Move to final folder
        outdir="bins/${file%.*}_$folder_type"
        mkdir -p "$outdir"
        mv "$tmp_outdir"/* "$outdir"
        rm -rf "$tmp_outdir"
    else
        echo "  ⚠️ No bytecode produced, skipping folder."
        rm -rf "$tmp_outdir"
    fi
done

# Convert all .bin files to .hex (and remove original .bin)
find "bins" -type f -name "*.bin" | while read -r file; do
    if [[ -s "$file" ]]; then
        newfile="${file%.bin}.hex"
        echo "Renaming: $file -> $newfile"
        mv "$file" "$newfile"
    else
        echo "  ⚠️ Skipping empty output: $file"
        rm -f "$file"
    fi
done

# Optionally, remove any stray empty .hex files (just in case)
find "bins" -type f -name "*.hex" -empty -delete
