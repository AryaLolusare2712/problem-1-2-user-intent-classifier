# Task 2 — User Intent Classifier

## Goal
Route natural-language questions over a large video corpus to the right retrieval method. The classifier predicts the primary intent, the relevant temporal scope, and a suggested retrieval strategy. It is a router, not the answer generator.

## Intent taxonomy
| Intent | Typical query | Scope | Retrieval treatment |
|---|---|---|---|
| `whole_video_summary` | “Summarize the whole video” | Whole video | Video metadata plus segment summaries or broad temporal coverage |
| `frame_visual_detail` | “What color is the car in this frame?” | Frame | Frame embeddings, timestamp and nearest-frame retrieval |
| `clip_temporal_event` | “When does the dog jump?” | Section/clip | Temporal segment index and event timestamps |
| `speech_audio_query` | “What did the host say about pricing?” | Section/clip | ASR transcript, speaker turns and timestamps |
| `on_screen_text_query` | “Read the scoreboard” | Frame/section | OCR text index with frame timestamps |
| `object_count_query` | “How many people are visible?” | Frame/section | Object detections/count metadata; temporal aggregation when required |
| `cross_video_comparison` | “Compare these two videos” | Multiple videos | Retrieve candidates from each video, align relevant segments, then compare |
| `general_semantic_search` | “Find footage of city traffic” | Adaptive | Hybrid semantic retrieval and reranking |

The supplied model emits one primary intent. Low-confidence predictions fall back to adaptive semantic retrieval and are marked for review. The mapping is deliberately extensible as the corpus and query logs reveal new intents.

## Lightweight architecture
Use a scikit-learn pipeline with word-level TF-IDF (unigrams and bigrams), character n-gram TF-IDF (3–5 characters), and class-balanced Logistic Regression. Sparse vectors are fast, inexpensive, interpretable, and work well as a baseline for short natural-language queries. No GPU, large transformer, or external API is required for inference. The fitted vectorizers and classifier are saved together in one versioned joblib artifact.

## Feature engineering
1. Normalize text with lowercase and Unicode accent normalization.
2. Word n-grams capture intent phrases such as “whole video”, “at what timestamp”, “what does it say”, and “compare videos”.
3. Character n-grams provide some robustness to spelling variations, typos and word-form differences.
4. Logistic Regression learns class-specific weights from sparse TF-IDF features and outputs confidence estimates.
5. Routing metadata maps each predicted intent to a retrieval scope and backend strategy.

Avoid hard-coded keyword-only rules as the primary classifier: query wording varies and classes overlap. A production system should add negated/ambiguous examples and use conversation context only when available and permitted.

## Dataset and evaluation
`data/intent_examples.csv` is a small seed dataset included to make the pipeline runnable. It is not a substitute for representative production data. Expand it with real and paraphrased queries sampled from the expected user population, balance classes, and label consistently. Keep near-duplicate paraphrases in the same data split to avoid leakage. Evaluate macro-F1, per-class precision/recall, confusion matrix, calibration and latency. The CLI prints a holdout report; the reported score is only a small-seed baseline and should not be presented as production performance.

## Run locally
```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/main.py train --csv data/intent_examples.csv --model artifacts/intent_classifier.joblib
python src/main.py predict --model artifacts/intent_classifier.joblib --query "When does the goal happen?"
```

## Deployment
- Package the model artifact, dependency lock/version, intent taxonomy and routing map together.
- Expose prediction through an internal HTTP endpoint (e.g. FastAPI) or call the Python classifier directly from the query router.
- At request time, validate query length, normalize text, run sparse TF-IDF and Logistic Regression, and return intent, confidence, scope and routing strategy.
- Use bounded CPU workers; cache repeated normalized queries if workload patterns justify it. The model is small and suitable for horizontal scaling.
- For confidence below a threshold calibrated on validation data, route to `general_semantic_search` or ask a clarifying question rather than forcing a brittle specialist route.
- Monitor p50/p95 latency, class distribution, confidence drift, fallback rate, and downstream retrieval success. Re-label failures and retrain on a versioned dataset.

## Limitations and extensions
Short queries can be ambiguous (“What happens here?”). Add context-aware features, timestamp/entity extraction, and multi-label routing if evaluation shows a query can require multiple indexes at once. For substantially larger or more varied corpora, compare this baseline against compact sentence embeddings or a small transformer using the same held-out evaluation.
