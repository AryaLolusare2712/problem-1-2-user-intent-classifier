# Problem 1.2 — User Intent Classifier

Lightweight query router for retrieval over a video corpus.

## Install and test
```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Train
```bash
python src/main.py train --csv data/intent_examples.csv --model artifacts/intent_classifier.joblib
```

## Predict
```bash
python src/main.py predict --model artifacts/intent_classifier.joblib --query "When does the goal happen?"
```

The included CSV is a small seed dataset for demonstrating the complete pipeline. Expand and evaluate on representative labelled queries before relying on it in production. See `docs/design.md` for taxonomy, feature engineering, evaluation, and deployment.
