#!/usr/bin/env python3
"""
Strip trailing numeric suffixes from patient name fields (family, given)
in all FHIR STU3 JSON files under data/fhir_stu3/.

Example: "Barton704" -> "Barton", "Alonso270" -> "Alonso"
"""

import json
import os
import re
import glob

FHIR_DIR = os.path.join(os.path.dirname(__file__), "fhir_stu3")


def strip_trailing_numbers(name: str) -> str:
    """Remove trailing digits from a name string."""
    return re.sub(r'\d+$', '', name)


def clean_names_in_resource(resource: dict) -> int:
    """Clean name fields in a single FHIR resource. Returns count of changes."""
    changes = 0
    if "name" not in resource:
        return 0

    for name_entry in resource["name"]:
        # Clean family name
        if "family" in name_entry:
            old = name_entry["family"]
            new = strip_trailing_numbers(old)
            if old != new:
                name_entry["family"] = new
                changes += 1

        # Clean given names
        if "given" in name_entry:
            for i, given in enumerate(name_entry["given"]):
                new = strip_trailing_numbers(given)
                if given != new:
                    name_entry["given"][i] = new
                    changes += 1

    return changes


def process_file(filepath: str) -> int:
    """Process a single FHIR JSON file. Returns count of name changes."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_changes = 0

    # Handle Bundle resources (most common in Synthea output)
    if data.get("resourceType") == "Bundle" and "entry" in data:
        for entry in data["entry"]:
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Patient":
                total_changes += clean_names_in_resource(resource)
    # Handle standalone Patient resources
    elif data.get("resourceType") == "Patient":
        total_changes += clean_names_in_resource(data)

    if total_changes > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return total_changes


def main():
    json_files = glob.glob(os.path.join(FHIR_DIR, "*.json"))
    print(f"Found {len(json_files)} JSON files in {FHIR_DIR}")

    files_modified = 0
    total_changes = 0

    for filepath in sorted(json_files):
        changes = process_file(filepath)
        if changes > 0:
            files_modified += 1
            total_changes += changes
            basename = os.path.basename(filepath)
            print(f"  ✓ {basename} ({changes} name field(s) cleaned)")

    print(f"\nDone! Modified {files_modified} files, {total_changes} total name changes.")


if __name__ == "__main__":
    main()
