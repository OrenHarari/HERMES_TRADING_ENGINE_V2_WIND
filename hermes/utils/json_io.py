"""Atomic JSON file IO for the learning layer.

Pure stdlib. Deterministic ordering (sort_keys=True). Atomic write via
write-temp-then-rename (os.replace is atomic on Windows and POSIX).
"""

import json
import os


def read_json(path, default=None):
    """Read JSON from `path`. Returns `default` if the file does not exist.

    Raises ValueError on parse error (caller may treat as 'corrupt config').
    """
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError("could not parse JSON {!s}: {!s}".format(path, exc))


def write_json_atomic(path, data):
    """Write `data` as JSON to `path` atomically.

    Creates parent directories if needed. Uses sort_keys=True for stable
    diffs and reproducible output. The temp file is in the same directory
    as the target so os.replace is a real atomic rename on the same volume.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, sort_keys=True, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
