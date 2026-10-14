"""Fast TF-IDF + Logistic Regression intent classifier for video-corpus retrieval."""
from __future__ import annotations

import os
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer as WordVectorizer

# Intent labels encode both the query's retrieval scope and the main operation.
INTENT_TO_SCOPE = {
    "whole_video_summary": "whole_video",
    "frame_visual_detail": "frame",
    "clip_temporal_event": "section_or_clip",
    "speech_audio_query": "section_or_clip",
    "on_screen_text_query": "frame_or_section",
    "object_count_query": "frame_or_section",
    "cross_video_comparison": "multiple_videos",
    "general_semantic_search": "adaptive",
}

INTENT_TO_ROUTING = {
    "whole_video_summary": "retrieve_video_level_metadata_and_sampled_or_all_segments",
    "frame_visual_detail": "retrieve_frame_embeddings_and_nearest_frames",
    "clip_temporal_event": "retrieve_temporal_segments_and_event_timestamps",
    "speech_audio_query": "search_asr_transcript_and_speaker_segments",
    "on_screen_text_query": "search_ocr_text_with_frame_timestamps",
    "object_count_query": "retrieve_object_detections_and_count_metadata",
    "cross_video_comparison": "retrieve_candidates_across_videos_then_compare",
    "general_semantic_search": "hybrid_search_then_adaptive_reranking",
}


def build_pipeline() -> Pipeline:
    """A small sparse-text model; no large neural model or external API required."""
    features = FeatureUnion([
        ("word", WordVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
            min_df=1, sublinear_tf=True, max_features=50000,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", lowercase=True, ngram_range=(3, 5),
            min_df=1, sublinear_tf=True, max_features=50000,
        )),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            max_iter=1500, class_weight="balanced", C=3.0,
            solver="lbfgs", random_state=42,
        )),
    ])


def train_model(csv_path: str, model_path: str, test_size: float = 0.25) -> dict[str, Any]:
    """Train and save a model from CSV columns query,intent."""
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split

    data = pd.read_csv(csv_path)
    if not {"query", "intent"}.issubset(data.columns):
        raise ValueError("Training CSV must contain 'query' and 'intent' columns.")
    data = data.dropna(subset=["query", "intent"])
    data["query"] = data["query"].astype(str).str.strip()
    data["intent"] = data["intent"].astype(str).str.strip()
    data = data[(data["query"] != "") & (data["intent"] != "")]
    if len(data) < 16 or data["intent"].nunique() < 2:
        raise ValueError("Provide at least 16 labelled queries and at least 2 intents.")

    counts = data["intent"].value_counts()
    n_test = max(1, int(round(len(data) * test_size)))
    n_train = len(data) - n_test
    stratify = data["intent"] if counts.min() >= 2 and n_test >= len(counts) and n_train >= len(counts) else None

    X_train, X_test, y_train, y_test = train_test_split(
        data["query"], data["intent"], test_size=test_size,
        random_state=42, stratify=stratify,
    )
    model = build_pipeline()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "holdout_accuracy": float(accuracy_score(y_test, predictions)),
        "classification_report": classification_report(
            y_test, predictions, zero_division=0, output_dict=True
        ),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "stratified_split": stratify is not None,
    }
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    joblib.dump({
        "model": model,
        "intent_to_scope": INTENT_TO_SCOPE,
        "intent_to_routing": INTENT_TO_ROUTING,
        "version": 1,
    }, model_path)
    return metrics


class IntentClassifier:
    def __init__(self, model_path: str):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.intent_to_scope = bundle.get("intent_to_scope", INTENT_TO_SCOPE)
        self.intent_to_routing = bundle.get("intent_to_routing", INTENT_TO_ROUTING)

    def predict(self, query: str, threshold: float = 0.42) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        probabilities = self.model.predict_proba([query.strip()])[0]
        best_index = int(probabilities.argmax())
        top_intent = str(self.model.classes_[best_index])
        confidence = float(probabilities[best_index])
        intent = top_intent if confidence >= threshold else "general_semantic_search"
        return {
            "query": query,
            "intent": intent,
            "model_top_intent": top_intent,
            "confidence": round(confidence, 4),
            "retrieval_scope": self.intent_to_scope.get(intent, "adaptive"),
            "routing_strategy": self.intent_to_routing.get(
                intent, INTENT_TO_ROUTING["general_semantic_search"]
            ),
            "needs_fallback_or_review": confidence < threshold,
        }
