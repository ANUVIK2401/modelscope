"""Reassemble the gzip archive emitted by the Cloud Run benchmark job."""

import argparse
import base64
import re
import tarfile
from pathlib import Path

RECORD = re.compile(r"^MODELSCOPE_ARTIFACT (\d+) ([A-Za-z0-9+/=]+)$")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_lines", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    chunks = {}
    for line in args.log_lines.read_text().splitlines():
        if match := RECORD.fullmatch(line.strip()):
            index = int(match[1])
            if index in chunks and chunks[index] != match[2]:
                raise SystemExit("Conflicting log entries. Filter to one job execution.")
            chunks[index] = match[2]
    if not chunks or set(chunks) != set(range(max(chunks) + 1)):
        raise SystemExit("Artifact log is missing one or more chunks.")
    try:
        with args.output.open("xb") as out:
            out.write(base64.b64decode("".join(chunks[i] for i in sorted(chunks)), validate=True))
        with tarfile.open(args.output, "r:gz") as artifact:
            artifact.getmembers()
    except (OSError, ValueError, tarfile.TarError) as exc:
        raise SystemExit(f"Could not validate archive: {exc}") from exc


if __name__ == "__main__":
    main()
