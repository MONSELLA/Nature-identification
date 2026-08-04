#!/usr/bin/env python3
"""
diagnose_artifact.py — characterise a damaged .jsonl artifact byte-by-byte.

Answers the question a line count alone CANNOT: was this file genuinely
truncated (a writer replaced it), or is it still full-size with its interior
replaced by NULL bytes (filesystem/storage data loss)? The two look
identical to `wc -l`, because NUL bytes contain no newlines — a 200 MB file
whose middle is nulled reports a handful of lines.

Distinguishing them matters:
  * truncated + rewritten -> a process opened it "w"; the old bytes are gone.
  * full size, NUL interior -> the writer succeeded and the STORAGE lost the
    data afterwards (unflushed blocks after a node crash, an NFS/cache
    failure, a filesystem-level fault). No amount of application-level
    locking prevents that, and it's diagnosed completely differently.

Prints the byte size, the line count, how many NUL bytes there are and
where, and how many records survive - plus a verdict.

Usage:
    python scripts/diagnose_artifact.py <path-to-.jsonl>
"""
import json
import sys
from pathlib import Path


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{n:,} B"
        n /= 1024


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"error: {path} does not exist")
        sys.exit(1)

    raw = path.read_bytes()
    size = len(raw)
    n_newlines = raw.count(b"\n")
    n_nulls = raw.count(b"\x00")

    # Where do the NULs sit? Contiguous runs are the signature of lost blocks
    # (filesystems zero whole blocks, typically 4 KiB, never single bytes).
    runs = []
    i = 0
    while i < len(raw):
        if raw[i] == 0:
            start = i
            while i < len(raw) and raw[i] == 0:
                i += 1
            runs.append((start, i - start))
        else:
            i += 1

    # How much still parses?
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
        rt = obj.get("record_type")
        if rt == "header":
            has_header = True
        elif rt == "footer":
            has_footer = True
        else:
            good += 1

    print(f"file:              {path}")
    print(f"size on disk:      {human(size)}  ({size:,} bytes)")
    print(f"newlines (wc -l):  {n_newlines:,}")
    print(f"NUL bytes:         {n_nulls:,}  ({n_nulls / size:.1%} of the file)"
          if size else "NUL bytes: 0")
    print(f"NUL runs:          {len(runs)}")
    for start, length in runs[:5]:
        print(f"    offset {start:,} .. {start + length:,}  ({human(length)})")
    if len(runs) > 5:
        print(f"    (+{len(runs) - 5} more runs)")
    print(f"parseable records: {good:,}   header={'Y' if has_header else 'N'} "
          f"footer={'Y' if has_footer else 'N'}   unparseable lines={bad}")

    # Average bytes per surviving record lets us estimate how many records the
    # file's SIZE could hold - the crux of the whole diagnosis.
    print()
    if good and n_nulls:
        bytes_per_rec = (size - n_nulls) / max(good, 1)
        implied = int(size / bytes_per_rec) if bytes_per_rec else 0
        print(f"VERDICT: file is {n_nulls / size:.0%} NUL bytes.")
        print(f"  At ~{bytes_per_rec:,.0f} bytes/record, this file's SIZE could hold "
              f"~{implied:,} records,")
        print(f"  but only {good:,} still parse. The rest was overwritten with zeros "
              f"IN PLACE.")
        print("  => This is STORAGE-level data loss (unflushed/lost blocks), NOT an")
        print("     application overwrite. A process opening the file 'w' would have")
        print("     produced a SMALL file, not a full-size one full of zeros.")
        print("  => Application-level file locking cannot prevent this. Look at the")
        print("     filesystem: node crash during/after the run, NFS or cache fault,")
        print("     quota/disk-full at flush time, or a storage outage.")
    elif n_nulls:
        print(f"VERDICT: {n_nulls:,} NUL bytes present but nothing parses - "
              f"file is largely destroyed.")
    else:
        print("VERDICT: no NUL bytes. The file is genuinely just short - consistent")
        print("  with a writer having replaced it (opened 'w'), or with inference")
        print("  never having written more than this. Compare the record count")
        print("  against what the run's log said it wrote.")


if __name__ == "__main__":
    main()
