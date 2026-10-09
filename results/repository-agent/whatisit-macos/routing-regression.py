import unittest
from whatisit_macos.engine import suggest

class MutationInflectionTests(unittest.TestCase):
    def test_mutation_inflections_abstain(self):
        for verb in ['changing', 'changed', 'deleting', 'deleted', 'removing', 'removed',
                     'erasing', 'erased', 'disabling', 'disabled', 'enabling', 'enabled',
                     'installing', 'installed', 'setting', 'writing', 'written']:
            with self.subTest(verb=verb):
                result = suggest(verb + ' battery cycle count', system='Darwin', which=lambda _: '/usr/bin/tool')
                self.assertEqual(result['status'], 'unsupported')
                self.assertIsNone(result['command'])

    def test_basic_inspection_still_works(self):
        result = suggest('show battery cycle count', system='Darwin', which=lambda _: '/usr/bin/tool')
        self.assertEqual(result['status'], 'suggestion')
