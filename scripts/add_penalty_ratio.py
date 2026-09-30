"""
add_penalty_ratio_v4.py

Adds the Stefan-model non-loop route penalty ratio to every 1D-PDPTW
instance JSON file in the v3 instance zip archive, producing a new v4
instance zip archive.

What it does
------------
For every *.json file found anywhere inside the input zip:
  1. Load the JSON (preserving key order).
  2. Insert the key:
         "non_loop_route_penalty_ratio": 0.5
     right after the "depot" key (falls back to inserting right before
     "json_format_version", or - if neither key exists - at the very
     start of the object, so the script never silently skips a file).
  3. If the key already exists, its value is left untouched by default
     (use --overwrite to force it to the given ratio).
  4. Re-serialize with the same formatting style as the source file
     (compact, no extra whitespace) and write it into the output zip
     at the same internal path.
Non-JSON files (e.g. the variant_v3_index.csv) are copied through
unchanged.

Usage
-----
    python add_penalty_ratio_v4.py <input_v3.zip> [output_v4.zip] \
        [--ratio 0.5] [--overwrite] [--dry-run]

Example
-------
    python add_penalty_ratio_v4.py instances_v3.zip instances_v4.zip
"""

import argparse
import json
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path


PENALTY_KEY = "non_loop_route_penalty_ratio"
DEFAULT_RATIO = 0.5


def _insert_penalty(obj: "OrderedDict", ratio: float, overwrite: bool) -> tuple[OrderedDict, str]:
    """
    Returns (new_ordered_dict, status) where status is one of:
        "added_after_depot", "added_before_version", "added_at_start",
        "already_present", "skipped_not_dict"
    """
    if not isinstance(obj, dict):
        return obj, "skipped_not_dict"

    keys = list(obj.keys())

    if PENALTY_KEY in keys:
        if not overwrite:
            return obj, "already_present"
        # Overwrite value in place, keep original position.
        new_obj = OrderedDict()
        for k in keys:
            new_obj[k] = ratio if k == PENALTY_KEY else obj[k]
        return new_obj, "overwritten"

    new_obj = OrderedDict()

    if "depot" in keys:
        for k in keys:
            new_obj[k] = obj[k]
            if k == "depot":
                new_obj[PENALTY_KEY] = ratio
        return new_obj, "added_after_depot"

    if "json_format_version" in keys:
        for k in keys:
            if k == "json_format_version":
                new_obj[PENALTY_KEY] = ratio
            new_obj[k] = obj[k]
        return new_obj, "added_before_version"

    # Neither anchor key present -> prepend (never skip the file).
    new_obj[PENALTY_KEY] = ratio
    for k in keys:
        new_obj[k] = obj[k]
    return new_obj, "added_at_start"


def process_zip(in_path: Path, out_path: Path, ratio: float,
                 overwrite: bool, dry_run: bool) -> None:
    status_counts: dict[str, int] = {}
    n_json = 0
    n_other = 0
    problems: list[str] = []

    with zipfile.ZipFile(in_path, "r") as zin:
        namelist = zin.namelist()

        zout = None
        if not dry_run:
            zout = zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED)

        try:
            for name in namelist:
                info = zin.getinfo(name)

                # Directories: just recreate the entry, nothing to touch.
                if name.endswith("/"):
                    if zout is not None:
                        zout.writestr(info, b"")
                    continue

                raw = zin.read(name)

                if name.lower().endswith(".json"):
                    n_json += 1
                    try:
                        data = json.loads(raw, object_pairs_hook=OrderedDict)
                    except json.JSONDecodeError as e:
                        problems.append(f"{name}: JSON parse error ({e})")
                        if zout is not None:
                            zout.writestr(info, raw)  # pass through unchanged
                        continue

                    new_data, status = _insert_penalty(data, ratio, overwrite)
                    status_counts[status] = status_counts.get(status, 0) + 1

                    if zout is not None:
                        # separators=(",", ": ") mirrors the compact-with-space
                        # style used in the source instance files.
                        new_raw = json.dumps(new_data, separators=(",", ": "))
                        zout.writestr(info, new_raw)
                else:
                    n_other += 1
                    if zout is not None:
                        zout.writestr(info, raw)
        finally:
            if zout is not None:
                zout.close()

    print(f"\nInput zip : {in_path}")
    if not dry_run:
        print(f"Output zip: {out_path}")
    else:
        print("(dry run - no output written)")

    print(f"\nJSON files found     : {n_json}")
    print(f"Non-JSON files copied: {n_other}")
    print("\nStatus breakdown:")
    for status, count in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {status:<24}: {count}")

    if problems:
        print(f"\n[!] {len(problems)} file(s) had problems:")
        for p in problems[:20]:
            print(f"  - {p}")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")

    already = status_counts.get("already_present", 0)
    if already and not overwrite:
        print(f"\nNote: {already} file(s) already had '{PENALTY_KEY}' and were "
              f"left untouched. Pass --overwrite to force-set the ratio.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_zip", type=Path, help="Path to the v3 instance zip archive.")
    parser.add_argument("output_zip", type=Path, nargs="?",
                         help="Path for the new v4 zip archive "
                              "(default: <input_stem>_v4.zip next to the input).")
    parser.add_argument("--ratio", type=float, default=DEFAULT_RATIO,
                         help=f"non_loop_route_penalty_ratio value to insert (default: {DEFAULT_RATIO}).")
    parser.add_argument("--overwrite", action="store_true",
                         help="If the key already exists in a file, overwrite its value instead of skipping.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Scan and report without writing an output zip.")
    args = parser.parse_args()

    if not args.input_zip.exists():
        print(f"Input zip not found: {args.input_zip}", file=sys.stderr)
        sys.exit(1)

    out_path = args.output_zip
    if out_path is None:
        stem = args.input_zip.stem
        # Replace a trailing _v3 / v3 with _v4 if present, else just append _v4.
        if stem.endswith("_v3"):
            new_stem = stem[: -len("_v3")] + "_v4"
        elif stem.endswith("v3"):
            new_stem = stem[: -len("v3")] + "v4"
        else:
            new_stem = stem + "_v4"
        out_path = args.input_zip.with_name(new_stem + ".zip")

    process_zip(args.input_zip, out_path, args.ratio, args.overwrite, args.dry_run)


if __name__ == "__main__":
    main()