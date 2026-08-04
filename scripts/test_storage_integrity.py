#!/usr/bin/env python3
"""
test_storage_integrity.py — write a large JSONL file the same way the
pipeline does, read it back, and check whether the bytes survived.

WHY: an artifact was found at full size (170 MB) with 97% of its interior
replaced by NUL bytes. Two competing explanations:
  (a) the storage lost written data (blocks never reached disk / were
      dropped by a network filesystem), or
  (b) something in this codebase wrote it that way.
Argument alone can't separate those. This reproduces the pipeline's exact
write pattern with NO pipeline code involved — no vLLM, no models, no
locking, just open/write/close in a loop — so if the file comes back
corrupted, the fault is provably below the application.

It also isolates whether fsync changes the outcome, which is directly
actionable: if the plain write corrupts and the fsync'd write doesn't, the
pipeline should be fsync'ing (and the storage is lying about durability on
close()).

Writes into the SAME directory as the real artifacts by default, because
filesystem behaviour is per-mount — testing in /tmp would prove nothing
about the results filesystem.

Usage:
    python scripts/test_storage_integrity.py \
        --dir /home/pmonserrat/code/results/vlm_pipeline/baseline/places/responses \
        --size_mb 200
    # then, to catch corruption that only shows up later:
    python scripts/test_storage_integrity.py --dir ... --verify_only
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

RECORD_PAD = 8000  # ~8.5 KB/record, matching the real artifacts' average


def _record(i):
    """Same shape and rough size as a real image record, so the write
    pattern (many medium-sized appends through Python's buffered writer)
    matches what phase_infer actually does."""
    return json.dumps({
        "record_type": "image",
        "image_path": f"/fake/img/{i}.jpg",
        "caption": "x" * RECORD_PAD,
        "objects": ["tree", "sky", "grass"],
        "idx": i,
    })


def write_file(path, n_records, do_fsync):
    t0 = time.time()
    with open(path, "w") as f:
        f.write(json.dumps({"record_type": "header", "test": True}) + "\n")
        for i in range(n_records):
            f.write(_record(i) + "\n")
        f.write(json.dumps({"record_type": "footer", "n": n_records}) + "\n")
        if do_fsync:
            # Force the data all the way to stable storage and RAISE if the
            # filesystem can't do it — the whole point being that a plain
            # close() on a network filesystem can return success while the
            # data is still only in a cache that may later be discarded.
            f.flush()
            os.fsync(f.fileno())
    return time.time() - t0


def verify(path, expect_records):
    raw = Path(path).read_bytes()
    size = len(raw)
    nulls = raw.count(b"\x00")
    good = bad = 0
    has_header = has_footer = False
    for line in raw.split(b"\n"):
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            bad += 1
            continue
        rt = obj.get("record_type") if isinstance(obj, dict) else None
        if rt == "header":
            has_header = True
        elif rt == "footer":
            has_footer = True
        else:
            good += 1
    ok = (nulls == 0 and good == expect_records and has_header and has_footer and bad == 0)
    print(f"    size={size:,}B  NULs={nulls:,} ({nulls / size:.1%})  "
          f"records={good:,}/{expect_records:,}  header={'Y' if has_header else 'N'} "
          f"footer={'Y' if has_footer else 'N'}  unparseable={bad}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True,
                    help="Directory to test — use the REAL results dir; filesystem "
                         "behaviour is per-mount, so testing elsewhere proves nothing.")
    ap.add_argument("--size_mb", type=int, default=200,
                    help="Approximate file size to write (default 200, ~the size of "
                         "a full places365 artifact).")
    ap.add_argument("--verify_only", action="store_true",
                    help="Skip writing; just re-verify files left by a previous run. "
                         "Use this MINUTES OR HOURS LATER to catch corruption that "
                         "appears after the fact rather than at write time.")
    ap.add_argument("--keep", action="store_true",
                    help="Don't delete the test files at the end.")
    args = ap.parse_args()

    d = Path(args.dir)
    if not d.is_dir():
        print(f"error: {d} is not a directory")
        sys.exit(1)

    n_records = max(1, (args.size_mb * 1_000_000) // (RECORD_PAD + 120))
    plain = d / "_storage_test_plain.jsonl"
    synced = d / "_storage_test_fsync.jsonl"

    failures = []
    if args.verify_only:
        for label, p in (("plain (no fsync)", plain), ("fsync'd", synced)):
            if p.exists():
                print(f"  {label}: {p.name}")
                if not verify(p, n_records):
                    failures.append(label)
            else:
                print(f"  {label}: {p.name} MISSING (was it deleted?)")
    else:
        print(f"Writing ~{args.size_mb} MB ({n_records:,} records) into {d}\n")
        for label, p, fs in (("plain (no fsync)", plain, False), ("fsync'd", synced, True)):
            dt = write_file(p, n_records, fs)
            print(f"  {label}: wrote in {dt:.1f}s")
            print("    immediately after close():")
            if not verify(p, n_records):
                failures.append(f"{label} (immediately)")
            # Drop this process's own view and re-open, so we're not just
            # reading back our own page cache from the same fd.
            time.sleep(2)
            print("    2s later, fresh open():")
            if not verify(p, n_records):
                failures.append(f"{label} (after 2s)")
            print()

    print("=" * 70)
    if failures:
        print("STORAGE FAULT REPRODUCED — corruption with NO pipeline code involved:")
        for f in failures:
            print(f"  - {f}")
        print("\nThis is proof the fault is below the application layer. Send this")
        print("output to your cluster admins with the directory path.")
    else:
        print("No corruption at write time. The storage handled this write correctly.")
        print("\nIMPORTANT: this does NOT clear the storage yet — the real artifact")
        print("was written over ~2.4 days, and a fault that only appears under long")
        print("runs, memory pressure, or a later server event would not show up in a")
        print("fast test. Re-run with --verify_only in an hour, and again tomorrow:")
        print(f"    python {sys.argv[0]} --dir {args.dir} --verify_only")
    if not args.keep and not args.verify_only and not failures:
        print("\n(pass --keep to retain the test files for --verify_only checks)")
        for p in (plain, synced):
            p.unlink(missing_ok=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
