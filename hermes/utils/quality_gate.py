"""Step 3 - Quality Gate / Code Integrity.

Deterministic AST-based scanner that flags engineering-rule violations:
  - import pandas / from pandas ...
  - import numpy  / from numpy ...
  - bare `.values` attribute access (pandas-style; `dict.values()` calls are
    NOT flagged - they are method calls and disambiguated via the AST).

Pure stdlib. No randomness. Deterministic ordering of findings.

Skip rules (pruned during os.walk):
  - .git, .hg, .svn, .tox, .nox, virtualenvs, caches, build artifacts, data/
  - any directory whose name starts with '.'
"""

import ast
import os

PROHIBITED_IMPORT_ROOTS = ("pandas", "numpy")

DEFAULT_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "venv",
        ".venv",
        "env",
        ".env",
        "virtualenv",
        ".virtualenv",
        "node_modules",
        "build",
        "dist",
        ".eggs",
        "site-packages",
        "data",
    }
)


def _iter_python_files(root, skip_dirs):
    """Yield .py paths under `root`, deterministically ordered, skipping
    directories whose basename is in `skip_dirs` or starts with '.'.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune in-place; sort for determinism.
        dirnames[:] = sorted(
            d for d in dirnames if d not in skip_dirs and not d.startswith(".")
        )
        for fname in sorted(filenames):
            if fname.endswith(".py"):
                yield os.path.join(dirpath, fname)


def _scan_source(path, source):
    """Return a list of finding dicts for one source string.

    Findings are dicts with keys: file, line, rule, message.
    """
    findings = []
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        findings.append(
            {
                "file": path,
                "line": getattr(exc, "lineno", 0) or 0,
                "rule": "syntax_error",
                "message": "could not parse file: {!s}".format(exc.msg),
            }
        )
        return findings

    # Pre-pass: identify Attribute nodes that are the `.func` of a Call so we
    # can distinguish method calls (dict.values()) from attribute access
    # (df.values). The pandas anti-pattern is bare attribute access; method
    # calls on stdlib types are fine.
    called_attr_ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            called_attr_ids.add(id(node.func))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in PROHIBITED_IMPORT_ROOTS:
                    findings.append(
                        {
                            "file": path,
                            "line": node.lineno,
                            "rule": "prohibited_import",
                            "message": "prohibited import: {!s}".format(alias.name),
                        }
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".", 1)[0] if mod else ""
            if root in PROHIBITED_IMPORT_ROOTS:
                findings.append(
                    {
                        "file": path,
                        "line": node.lineno,
                        "rule": "prohibited_import_from",
                        "message": "prohibited from-import: from {!s}".format(mod),
                    }
                )
        elif isinstance(node, ast.Attribute):
            if node.attr == "values" and id(node) not in called_attr_ids:
                findings.append(
                    {
                        "file": path,
                        "line": node.lineno,
                        "rule": "values_attribute_usage",
                        "message": (
                            "prohibited '.values' attribute access "
                            "(pandas-style anti-pattern)"
                        ),
                    }
                )

    return findings


def _check_file(path):
    """Read and scan a single file. Returns list of finding dicts."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            source = fh.read()
    except OSError as exc:
        return [
            {
                "file": path,
                "line": 0,
                "rule": "unreadable_file",
                "message": "could not read file: {!s}".format(exc),
            }
        ]
    return _scan_source(path, source)


def scan_paths(roots, skip_dirs=None):
    """Scan one or more root directories.

    Returns a list of finding dicts, sorted deterministically by
    (file, line, rule).
    """
    if skip_dirs is None:
        skip_dirs = DEFAULT_SKIP_DIRS
    findings = []
    for root in roots:
        if not os.path.isdir(root):
            # Allow scanning a single file as well.
            if os.path.isfile(root) and root.endswith(".py"):
                findings.extend(_check_file(root))
            continue
        for path in _iter_python_files(root, skip_dirs):
            findings.extend(_check_file(path))
    findings.sort(key=lambda f: (f["file"], f["line"], f["rule"]))
    return findings


def format_findings(findings):
    """Render findings as a deterministic, human-readable string."""
    if not findings:
        return "OK: 0 findings"
    lines = ["{!s} finding(s):".format(len(findings))]
    for f in findings:
        lines.append(
            "  {file}:{line}: [{rule}] {message}".format(
                file=f["file"], line=f["line"], rule=f["rule"], message=f["message"]
            )
        )
    return "\n".join(lines)
