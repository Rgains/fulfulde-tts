import unicodedata
import unittest

from fub_tts.text import analyze_text, normalize_text


class TextNormalizationTests(unittest.TestCase):
    def test_normalization_is_nfc_and_preserves_fulfulde(self) -> None:
        source = "  Ɓiŋgel\tmaako  jooɗi  bee  ƴiɗde.  "
        result = normalize_text(source)
        self.assertEqual(result, "Ɓiŋgel maako jooɗi bee ƴiɗde.")
        self.assertTrue(unicodedata.is_normalized("NFC", result))
        self.assertIn("joo", result)
        self.assertIn("ɗ", result)
        self.assertIn("ƴ", result)
        self.assertIn("ŋ", result)

    def test_analysis_flags_without_changing_text(self) -> None:
        source = "NGO waɗi ɗum 2 laawol."
        analysis = analyze_text(source)
        self.assertEqual(analysis.normalized, source)
        self.assertIn("contains_number", analysis.flags)
        self.assertIn("possible_abbreviation", analysis.flags)
        self.assertNotIn("unknown_script", analysis.flags)

    def test_modifier_apostrophe_is_preserved_and_not_an_unknown_script(self) -> None:
        source = "Ndikka yiigo e wiʼeego."
        analysis = analyze_text(source)
        self.assertEqual(analysis.normalized, source)
        self.assertNotIn("unknown_script", analysis.flags)


if __name__ == "__main__":
    unittest.main()
