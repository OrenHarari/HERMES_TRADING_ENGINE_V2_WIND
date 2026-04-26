"""Tests for Prompt 1 / Step 3 - Quality Gate / Code Integrity."""

import os
import shutil
import tempfile
import unittest

from hermes.utils.quality_gate import (
    DEFAULT_SKIP_DIRS,
    PROHIBITED_IMPORT_ROOTS,
    format_findings,
    scan_paths,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestQualityGateOnRealProject(unittest.TestCase):
    """The actual codebase must be clean: no pandas, no numpy, no .values."""

    def test_hermes_package_is_clean(self):
        findings = scan_paths([os.path.join(REPO_ROOT, "hermes")])
        self.assertEqual(
            findings, [], msg="hermes/ has findings:\n" + format_findings(findings)
        )

    def test_tests_package_is_clean(self):
        findings = scan_paths([os.path.join(REPO_ROOT, "tests")])
        self.assertEqual(
            findings, [], msg="tests/ has findings:\n" + format_findings(findings)
        )

    def test_full_project_is_clean(self):
        findings = scan_paths([REPO_ROOT])
        self.assertEqual(
            findings, [], msg="project has findings:\n" + format_findings(findings)
        )


class TestQualityGateDetectsOffenders(unittest.TestCase):
    """Synthetic offenders in a tmp dir must be detected."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="hermes_qg_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, relpath, source):
        path = os.path.join(self.tmp, relpath)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(source)
        return path

    def test_detects_import_pandas(self):
        self._write("bad_pd.py", "import pandas\n")
        findings = scan_paths([self.tmp])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["rule"], "prohibited_import")
        self.assertIn("pandas", findings[0]["message"])

    def test_detects_import_pandas_as_alias(self):
        self._write("bad_pd_alias.py", "import pandas as pd\n")
        findings = scan_paths([self.tmp])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["rule"], "prohibited_import")

    def test_detects_from_pandas_import(self):
        self._write("bad_from_pd.py", "from pandas import DataFrame\n")
        findings = scan_paths([self.tmp])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["rule"], "prohibited_import_from")

    def test_detects_import_numpy(self):
        self._write("bad_np.py", "import numpy as np\n")
        findings = scan_paths([self.tmp])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["rule"], "prohibited_import")
        self.assertIn("numpy", findings[0]["message"])

    def test_detects_from_numpy_submodule(self):
        self._write("bad_np_sub.py", "from numpy.linalg import inv\n")
        findings = scan_paths([self.tmp])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["rule"], "prohibited_import_from")

    def test_detects_bare_values_attribute(self):
        # pandas-style: df.values (no call). MUST be flagged.
        src = "def f(df):\n    return df.values\n"
        self._write("bad_values.py", src)
        findings = scan_paths([self.tmp])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["rule"], "values_attribute_usage")

    def test_does_not_flag_dict_values_method_call(self):
        # stdlib pattern: d.values(). MUST NOT be flagged.
        src = (
            "def f(d):\n"
            "    return list(d.values())\n"
            "def g(d):\n"
            "    for v in d.values():\n"
            "        yield v\n"
        )
        self._write("ok_dict_values.py", src)
        findings = scan_paths([self.tmp])
        self.assertEqual(findings, [])

    def test_detects_multiple_findings_sorted(self):
        self._write(
            "many.py",
            "import pandas\n"
            "import numpy\n"
            "def f(df):\n"
            "    return df.values\n",
        )
        findings = scan_paths([self.tmp])
        self.assertEqual(len(findings), 3)
        # Sorted by (file, line, rule).
        lines = [f["line"] for f in findings]
        self.assertEqual(lines, sorted(lines))

    def test_skips_default_skip_dirs(self):
        # An offender inside __pycache__ or .git must NOT be detected.
        for skip in ("__pycache__", ".git", "venv", "data"):
            self.assertIn(skip, DEFAULT_SKIP_DIRS) if skip == "__pycache__" or skip == "data" else None
            self._write(os.path.join(skip, "bad.py"), "import pandas\n")
        findings = scan_paths([self.tmp])
        self.assertEqual(findings, [])

    def test_skips_dotted_dirs(self):
        # Any directory starting with '.' is skipped.
        self._write(".hidden/bad.py", "import numpy\n")
        findings = scan_paths([self.tmp])
        self.assertEqual(findings, [])

    def test_clean_file_produces_no_findings(self):
        self._write(
            "clean.py",
            "def f(x):\n"
            "    return x + 1\n"
            "def g(d):\n"
            "    return list(d.values())\n",
        )
        findings = scan_paths([self.tmp])
        self.assertEqual(findings, [])

    def test_deterministic_repeated_scan(self):
        self._write("a.py", "import pandas\n")
        self._write("b.py", "import numpy\n")
        self._write("c.py", "def f(df): return df.values\n")
        first = scan_paths([self.tmp])
        second = scan_paths([self.tmp])
        third = scan_paths([self.tmp])
        self.assertEqual(first, second)
        self.assertEqual(second, third)

    def test_handles_syntax_error_gracefully(self):
        self._write("broken.py", "def f(:\n")
        findings = scan_paths([self.tmp])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["rule"], "syntax_error")

    def test_string_literal_pandas_not_flagged(self):
        # The literal string "pandas" in a string constant must not be flagged
        # as an import; only AST Import/ImportFrom nodes are flagged.
        self._write(
            "literals.py",
            "MESSAGE = 'pandas and numpy are not allowed'\n"
            "def f():\n"
            "    return 'do not use df.values'\n",
        )
        findings = scan_paths([self.tmp])
        self.assertEqual(findings, [])


class TestQualityGateConstants(unittest.TestCase):
    def test_prohibited_roots_match_spec(self):
        self.assertIn("pandas", PROHIBITED_IMPORT_ROOTS)
        self.assertIn("numpy", PROHIBITED_IMPORT_ROOTS)


if __name__ == "__main__":
    unittest.main()
