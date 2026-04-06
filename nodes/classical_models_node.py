"""
Classical models node — runs RF and KNN inference on the live feature vector.

Assembles features from analyst signals (technical, fundamental, macro, sentiment)
and data (sector), loads the trained per-horizon models, and writes predictions
into analyst_signals so the synthesizer can incorporate them.
"""

import os

import joblib

from shared.state_schema import BerkshireState
from shared.feature_engineering import FEATURE_ORDER, SECTORS, assemble_feature_vector
from shared.horizon import normalize_horizon, horizon_label

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def _load(filename: str):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _gather_features(state: BerkshireState) -> dict:
    """Pull computed features from analyst signals and raw data."""
    signals = state.get("analyst_signals", {})
    data = state.get("data", {})

    features = {}

    # Technical features (7)
    tech = signals.get("technical", {})
    if isinstance(tech, dict):
        for k, v in (tech.get("features") or {}).items():
            features[k] = v

    # Fundamental features (5)
    fund = signals.get("fundamental", {})
    if isinstance(fund, dict):
        for k, v in (fund.get("features") or {}).items():
            features[k] = v

    # Macro features (2)
    macro = signals.get("macro", {})
    if isinstance(macro, dict):
        for k, v in (macro.get("features") or {}).items():
            if k in ("sector_performance", "market_trend"):
                features[k] = v

    # Sentiment features (2)
    sent = signals.get("sentiment", {})
    if isinstance(sent, dict):
        sent_features = sent.get("features") or {}
        features["sentiment_score"] = sent_features.get("sentiment_score")
        features["news_volume"] = sent_features.get("news_volume")

    # Sector one-hot (9)
    company_info = data.get("company_info", {})
    sector = company_info.get("sector", "") if isinstance(company_info, dict) else ""
    for s in SECTORS:
        key = f"sector_{s.lower().replace(' ', '_')}"
        features[key] = 1 if sector and s.lower() == sector.lower() else 0

    return features


def classical_models_node(state: BerkshireState):
    """Load trained RF/KNN models and predict on live features."""
    ticker = state.get("ticker", "UNKNOWN")
    horizon = normalize_horizon(state.get("horizon", "swing"))
    horizon_lbl = horizon_label(horizon)

    features = _gather_features(state)
    vector = assemble_feature_vector(features)

    imputer = _load(f"imputer_{horizon}.pkl")
    rf = _load(f"rf_{horizon}.pkl")
    knn = _load(f"knn_{horizon}.pkl")
    scaler = _load(f"scaler_{horizon}.pkl")

    results = {}

    if imputer is not None:
        X = imputer.transform([vector])

        if rf is not None:
            rf_pred = rf.predict(X)[0]
            rf_proba = dict(zip(rf.classes_, [round(float(p), 4) for p in rf.predict_proba(X)[0]]))
            results["rf"] = {
                "prediction": rf_pred,
                "probabilities": rf_proba,
            }
            print(f"\n[Classical] RF ({horizon}): {rf_pred}  {rf_proba}")

        if knn is not None and scaler is not None:
            X_scaled = scaler.transform(X)
            knn_pred = knn.predict(X_scaled)[0]
            knn_proba = dict(zip(knn.classes_, [round(float(p), 4) for p in knn.predict_proba(X_scaled)[0]]))
            results["knn"] = {
                "prediction": knn_pred,
                "probabilities": knn_proba,
            }
            print(f"[Classical] KNN ({horizon}): {knn_pred}  {knn_proba}")
    else:
        print(f"[Classical] No imputer found for horizon '{horizon}'. Skipping inference.")

    if not results:
        print(f"[Classical] No models available for {horizon}. Skipping.")

    feature_coverage = sum(1 for f in FEATURE_ORDER if features.get(f) is not None)
    print(f"[Classical] Feature coverage: {feature_coverage}/{len(FEATURE_ORDER)}")

    return {
        "analyst_signals": {
            "classical_models": {
                "horizon": horizon,
                "horizon_label": horizon_lbl,
                "rf": results.get("rf"),
                "knn": results.get("knn"),
                "feature_coverage": feature_coverage,
                "total_features": len(FEATURE_ORDER),
            }
        }
    }
