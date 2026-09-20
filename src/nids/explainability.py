"""
Stage 13 — Explainability & Evidence Chain.

Produces analyst-facing explanations combining TreeSHAP feature attributions
(RF and XGBoost), attack-graph path evidence for a flagged session's token
sequence, and a consolidated natural-language evidence-chain report.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import networkx as nx
import numpy as np

from config import NOVEL_EDGE_WEIGHT, RISK_SUSPICIOUS_THRESH


def compute_treeshap_rf(rf, X_sample: np.ndarray, feature_names: Sequence[str]):
    import shap

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


def compute_treeshap_xgb(xgb_clf, X_sample: np.ndarray, feature_names: Sequence[str]):
    import shap

    # If xgb_clf is a _GlobalLabelXGBClassifier (models.py) wrapping a
    # locally-re-indexed booster, TreeExplainer needs the raw booster;
    # shap_values columns then correspond to xgb_clf.classes_ (local
    # order), not necessarily the full global 0..K-1 class space.
    model_for_shap = getattr(xgb_clf, "booster_", xgb_clf)
    explainer = shap.TreeExplainer(model_for_shap)
    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values


def extract_graph_path_evidence(session_tokens: List[str], G: nx.DiGraph) -> List[dict]:
    """For each consecutive token transition, report the graph edge weight and novelty."""
    evidence = []
    for i in range(len(session_tokens) - 1):
        src, dst = session_tokens[i], session_tokens[i + 1]
        has_edge = G.has_edge(src, dst)
        weight = G[src][dst]["weight"] if has_edge else NOVEL_EDGE_WEIGHT
        evidence.append({
            "from": src,
            "to": dst,
            "weight": weight,
            "novel": not has_edge,
        })
    return evidence


def _top_shap_features(shap_values, feature_names: Sequence[str], class_idx: int, top_n: int = 5):
    """Return (feature_name, mean_shap_value) sorted by |mean_shap_value| descending."""
    if isinstance(shap_values, list):
        # Multi-class TreeSHAP: list of (n_samples, n_features) arrays, one per class.
        idx = min(class_idx, len(shap_values) - 1)
        values = np.asarray(shap_values[idx])
    else:
        values = np.asarray(shap_values)
        if values.ndim == 3:
            idx = min(class_idx, values.shape[2] - 1)
            values = values[:, :, idx]

    mean_abs = np.abs(values).mean(axis=0)
    mean_signed = values.mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    return [(feature_names[i], float(mean_signed[i])) for i in order]


def build_evidence_report(
    session_id: str, session_info: dict, risk_score: float,
    sp_score: float, tc_score: float, gw_score: float,
    shap_values_rf, rf_feature_names: Sequence[str],
    path_evidence: List[dict],
) -> str:
    """Formatted multi-line analyst evidence-chain report for a flagged session."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"EVIDENCE CHAIN REPORT — Session {session_id}")
    lines.append("=" * 70)
    lines.append(f"True/Assigned Label : {session_info.get('label', 'UNKNOWN')}")
    lines.append(f"Risk Score (Risk_t) : {risk_score:.4f}")
    lines.append(f"Pattern Match (SP_t): {sp_score:.4f}")
    lines.append(f"Graph Consistency (TC_t): {tc_score:.4f}")
    lines.append(f"Graph Centrality (G_w)  : {gw_score:.4f}")
    lines.append("")

    lines.append("Top-5 SHAP Feature Attributions (Random Forest):")
    try:
        top_feats = _top_shap_features(shap_values_rf, list(rf_feature_names), class_idx=1)
        for name, val in top_feats:
            arrow = "▲" if val >= 0 else "▼"
            lines.append(f"  {arrow} {name}: {val:+.4f}")
    except Exception as exc:  # pragma: no cover - defensive formatting path
        lines.append(f"  (SHAP attribution unavailable: {exc})")
    lines.append("")

    tokens = session_info.get("tokens", [])
    lines.append(f"Token Sequence ({len(tokens)} events):")
    lines.append("  " + " -> ".join(tokens))
    lines.append("")

    high_weight_transitions = [e for e in path_evidence if e["weight"] > 0.5]
    lines.append(f"High-Weight Graph Transitions ({len(high_weight_transitions)}):")
    for e in high_weight_transitions:
        novel_tag = " [NOVEL]" if e["novel"] else ""
        lines.append(f"  {e['from']} -> {e['to']} (w={e['weight']:.3f}){novel_tag}")
    lines.append("")

    if risk_score >= RISK_SUSPICIOUS_THRESH:
        action = "BLOCK & ESCALATE to SOC Tier-2 analyst for manual triage."
    elif risk_score >= 0.35:
        action = "MONITOR: flag for correlation with subsequent sessions from this host pair."
    else:
        action = "NO ACTION: consistent with benign traffic baseline."
    lines.append(f"Recommended Action: {action}")
    lines.append("=" * 70)

    return "\n".join(lines)
