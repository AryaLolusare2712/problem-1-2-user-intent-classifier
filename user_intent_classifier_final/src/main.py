"""CLI for training and using the user intent classifier."""
import argparse
import json
import os
import sys

from classifier import IntentClassifier, train_model


def main():
    parser = argparse.ArgumentParser(description="Classify queries for video-corpus retrieval.")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="Train from a labelled query CSV.")
    train_parser.add_argument("--csv", default="data/intent_examples.csv")
    train_parser.add_argument("--model", default="artifacts/intent_classifier.joblib")
    train_parser.add_argument("--test-size", type=float, default=0.25)

    predict_parser = commands.add_parser("predict", help="Classify one query.")
    predict_parser.add_argument("--model", default="artifacts/intent_classifier.joblib")
    predict_parser.add_argument("--threshold", type=float, default=0.42)
    predict_parser.add_argument("--query", required=True)

    args = parser.parse_args()
    if args.command == "train":
        if not 0 < args.test_size < 1:
            parser.error("--test-size must be between 0 and 1")
        try:
            metrics = train_model(args.csv, args.model, args.test_size)
        except Exception as exc:
            print(f"Training failed: {exc}", file=sys.stderr)
            raise SystemExit(1)
        print(json.dumps({
            "model_path": args.model,
            "holdout_accuracy": round(metrics["holdout_accuracy"], 4),
            "train_rows": metrics["train_rows"],
            "test_rows": metrics["test_rows"],
            "stratified_split": metrics["stratified_split"],
            "classification_report": metrics["classification_report"],
        }, indent=2))
    else:
        try:
            result = IntentClassifier(args.model).predict(args.query, args.threshold)
        except Exception as exc:
            print(f"Prediction failed: {exc}", file=sys.stderr)
            raise SystemExit(1)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
