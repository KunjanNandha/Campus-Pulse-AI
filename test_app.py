"""Regression checks for abbreviated, mixed, and sarcastic feedback."""
import unittest
from datetime import datetime

from streamlit.testing.v1 import AppTest


class FeedbackRulesTest(unittest.TestCase):
    def analyzed(self, comments):
        app = AppTest.from_file("app.py", default_timeout=40).run()
        self.assertFalse(app.exception)
        app.session_state["rows"] = [
            {"feedback": text, "service": service, "timestamp": datetime.now(), "rating": None}
            for text, service in comments
        ]
        app.run()
        self.assertFalse(app.exception)
        table = next(frame.value for frame in app.dataframe if "scoring_text" in frame.value.columns)
        return table

    def test_abbreviation_and_opposite_clauses(self):
        table = self.analyzed([
            ("the clg is nt great but food is very good", "Canteen"),
            ("The canteen food is good but the price is terrible.", "Canteen"),
        ])
        first = table[table.feedback.eq("the clg is nt great but food is very good")].iloc[0]
        second = table[table.feedback.eq("The canteen food is good but the price is terrible.")].iloc[0]
        self.assertEqual(first["sentiment"], "Mixed")
        self.assertEqual(first["scoring_text"], "the college is not great but food is very good")
        self.assertIn("Campus / General campus: Negative", first["aspects"])
        self.assertIn("Canteen / Food quality: Positive", first["aspects"])
        self.assertTrue(table["derived_stars"].isna().all())
        self.assertEqual(second["sentiment"], "Mixed")
        self.assertIn("Canteen / Price: Negative", second["aspects"])

    def test_sarcasm_requires_a_complaint(self):
        table = self.analyzed([
            ("Oh great, the Wi-Fi is down again.", "Wi-Fi"),
            ("The Wi-Fi is great.", "Wi-Fi"),
        ])
        complaint = table[table.feedback.eq("Oh great, the Wi-Fi is down again.")].iloc[0]
        praise = table[table.feedback.eq("The Wi-Fi is great.")].iloc[0]
        self.assertEqual(complaint["sentiment"], "Negative")
        self.assertTrue(complaint["sarcasm_adjusted"])
        self.assertIn("Wi-Fi / Connectivity: Negative", complaint["aspects"])
        self.assertEqual(praise["sentiment"], "Positive")
        self.assertFalse(praise["sarcasm_adjusted"])

    def test_multiple_sentences_and_clause_wording(self):
        cases = [
            ("College facilities are poor; canteen food is excellent.", "Campus / General campus: Negative", "Canteen / Food quality: Positive"),
            ("The bus is late. The driver is polite.", "Transport / Punctuality: Negative", "Transport / Driver behaviour: Positive"),
            ("The Wi-Fi is slow, but the library is peaceful.", "Wi-Fi / Speed: Negative", "Library / Noise: Positive"),
            ("The library is quiet but there are no seats.", "Library / Noise: Positive", "Library / Seating: Negative"),
        ]
        table = self.analyzed([(text, "Other") for text, _, _ in cases])
        for text, negative_or_positive, opposite in cases:
            with self.subTest(text=text):
                row = table[table.feedback.eq(text)].iloc[0]
                self.assertEqual(row["sentiment"], "Mixed")
                self.assertIn(negative_or_positive, row["aspects"])
                self.assertIn(opposite, row["aspects"])


if __name__ == "__main__":
    unittest.main()
