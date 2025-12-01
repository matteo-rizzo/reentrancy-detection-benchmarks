import os
import re
from collections import Counter

from numpy import sort

def extract_pragma_version(content):
    """
    Extracts the raw pragma version (e.g. '^0.4.25', '>=0.6.1').
    """
    pattern = r'pragma\s+solidity\s+([^;]+);'
    match = re.search(pattern, content)
    return match.group(1).strip() if match else None

def normalize_version(raw):
    """
    Removes any leading non-numeric characters so that:
    '^0.4.25'  -> '0.4.25'
    '>=0.6.1'  -> '0.6.1'
    '~0.5.0'   -> '0.5.0'
    """
    match = re.search(r'\d+(\.\d+){0,2}', raw)
    return match.group(0) if match else raw

def extract_major_version(normalized):
    """
    Extracts major.minor from a normalized version:
    '0.4.25' -> '0.4'
    '0.6.1'  -> '0.6'
    """
    match = re.match(r'(\d+)\.(\d+)', normalized)
    if match:
        return f"{match.group(1)}.{match.group(2)}"
    return "(unknown)"

def count_pragmas_in_folder(folder):
    exact_counter = Counter()
    major_counter = Counter()

    for filename in os.listdir(folder):
        if filename.endswith(".sol"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            raw = extract_pragma_version(content)
            if raw:
                normalized = normalize_version(raw)
                major = extract_major_version(normalized)

                exact_counter[normalized] += 1
                major_counter[major] += 1
            else:
                exact_counter["(no pragma)"] += 1
                major_counter["(no pragma)"] += 1

    return exact_counter, major_counter



if __name__ == "__main__":
    folder = "../benchmarks/aggregated-benchmark/raw/reentrant"
    exact, major = count_pragmas_in_folder(folder)

    print("Contracts per normalized pragma version:")
    for pragma, count in sorted(exact.items()):
        print(f"{pragma}: {count}")

    print("\nContracts per major version:")
    for version, count in sorted(major.items()):
        print(f"{version}: {count}")

