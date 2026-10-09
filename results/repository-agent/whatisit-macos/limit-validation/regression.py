import tempfile
import unittest
from pathlib import Path
from whatisit_macos import retrieval

class SearchLimitTests(unittest.TestCase):
    def test_invalid_limits_raise_value_error_before_database_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'absent.sqlite3'
            for limit in [True, False, 1.5, '3', None, 0, 11]:
                with self.subTest(limit=limit), self.assertRaises(ValueError):
                    retrieval.search(path, 'power', limit=limit)
            self.assertFalse(path.exists())

    def test_valid_integer_limits_still_search(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'docs.sqlite3'
            retrieval.build_index(path, [retrieval.Manual('pmset', 'test manual', '/usr/bin/pmset', '27.0.1', '2026-10-09', 'power assertions')])
            for limit in [1, 3, 10]:
                with self.subTest(limit=limit):
                    self.assertEqual(retrieval.search(path, 'power', limit=limit)[0]['tool'], 'pmset')
