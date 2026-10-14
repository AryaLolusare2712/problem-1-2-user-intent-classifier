import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from classifier import IntentClassifier, train_model


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestIntentClassifier(unittest.TestCase):
    def test_training_and_prediction(self):
        with tempfile.TemporaryDirectory() as temp:
            model_path = os.path.join(temp, "model.joblib")
            metrics = train_model(
                os.path.join(ROOT, "data", "intent_examples.csv"),
                model_path,
                test_size=0.25,
            )
            self.assertTrue(os.path.isfile(model_path))
            self.assertGreaterEqual(metrics["holdout_accuracy"], 0.0)
            classifier = IntentClassifier(model_path)
            result = classifier.predict("Summarize the entire video")
            self.assertIn(result["intent"], {
                "whole_video_summary", "general_semantic_search"
            })
            self.assertIn("retrieval_scope", result)
            self.assertIn("routing_strategy", result)

    def test_empty_query_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            model_path = os.path.join(temp, "model.joblib")
            train_model(
                os.path.join(ROOT, "data", "intent_examples.csv"),
                model_path,
                test_size=0.25,
            )
            classifier = IntentClassifier(model_path)
            with self.assertRaises(ValueError):
                classifier.predict("   ")


if __name__ == "__main__":
    unittest.main()
