import unittest

from character_gpt.train import diagnose_fit


class TrainingDiagnosticsTests(unittest.TestCase):
    def test_detects_overfitting_risk(self):
        history = [
            {"train": 2.0, "validation": 2.1},
            {"train": 1.2, "validation": 1.9},
            {"train": 0.8, "validation": 2.2},
        ]
        self.assertIn("Overfitting", diagnose_fit(history, vocab_size=30))

    def test_detects_underfitting_risk(self):
        history = [
            {"train": 3.3, "validation": 3.4},
            {"train": 3.32, "validation": 3.4},
            {"train": 3.31, "validation": 3.42},
        ]
        self.assertIn("Underfitting", diagnose_fit(history, vocab_size=30))

    def test_waits_for_enough_evaluations(self):
        history = [{"train": 3.0, "validation": 3.1}]
        self.assertIn("Not enough", diagnose_fit(history, vocab_size=30))


if __name__ == "__main__":
    unittest.main()
