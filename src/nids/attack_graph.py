"""
Stages 9-10 — Attack-State Graph & Temporal Consistency Scoring.

Builds a token-transition graph from real training sessions with
exponential-moving-average (EMA) edge weights, then scores held-out
sessions by how consistent their observed token transitions are with the
learned attack-state graph (TC_t), combined with a betweenness-centrality
based graph-weight score (G_w).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import networkx as nx

from config import EMA_RHO, LAMBDA_DECAY, NOVEL_EDGE_WEIGHT, VOCAB


def _new_transition_weights(sessions) -> Dict[tuple, float]:
    """Per-batch empirical transition probabilities: count(src->dst) / total_outgoing(src)."""
    counts: Dict[tuple, int] = {}
    outgoing_totals: Dict[str, int] = {}

    for s in sessions:
        tokens = s["tokens"] if isinstance(s, dict) else s
        for i in range(len(tokens) - 1):
            src, dst = tokens[i], tokens[i + 1]
            counts[(src, dst)] = counts.get((src, dst), 0) + 1
            outgoing_totals[src] = outgoing_totals.get(src, 0) + 1

    weights = {}
    for (src, dst), c in counts.items():
        weights[(src, dst)] = c / outgoing_totals[src]
    return weights


def build_attack_graph(sessions_dict: Dict[str, dict], rho: float = EMA_RHO) -> nx.DiGraph:
    """All VOCAB tokens are always present as nodes, even with zero edges."""
    G = nx.DiGraph()
    for token in VOCAB:
        G.add_node(token)

    new_weights = _new_transition_weights(list(sessions_dict.values()))
    for (src, dst), w_new in new_weights.items():
        # First observation: EMA initializes directly to the observed rate.
        G.add_edge(src, dst, weight=w_new)

    return G


def update_graph_ema(G: nx.DiGraph, new_sessions, rho: float = EMA_RHO) -> nx.DiGraph:
    """
    Incremental EMA update for streaming use: W_t = rho*W_old + (1-rho)*W_new.
    Any node that appears as a source in the new batch has ALL of its
    outgoing edges re-blended (edges not observed in this batch decay
    toward zero rather than staying frozen), matching a proper streaming
    re-estimate of that node's outgoing-transition distribution.
    """
    counts: Dict[tuple, int] = {}
    outgoing_totals: Dict[str, int] = {}
    for s in new_sessions:
        tokens = s["tokens"] if isinstance(s, dict) else s
        for i in range(len(tokens) - 1):
            src, dst = tokens[i], tokens[i + 1]
            counts[(src, dst)] = counts.get((src, dst), 0) + 1
            outgoing_totals[src] = outgoing_totals.get(src, 0) + 1

    touched_sources = set(outgoing_totals.keys())

    # Decay edges from touched sources that were NOT re-observed this batch.
    for src in touched_sources:
        if src in G:
            for dst in list(G.successors(src)):
                if (src, dst) not in counts:
                    w_old = G[src][dst]["weight"]
                    G[src][dst]["weight"] = rho * w_old

    for (src, dst), c in counts.items():
        w_new = c / outgoing_totals[src]
        if G.has_edge(src, dst):
            w_old = G[src][dst]["weight"]
            G[src][dst]["weight"] = rho * w_old + (1 - rho) * w_new
        else:
            if src not in G:
                G.add_node(src)
            if dst not in G:
                G.add_node(dst)
            G.add_edge(src, dst, weight=w_new)
    return G


def compute_tc_score(
    session_tokens: List[str], G: nx.DiGraph, timestamps: Optional[List[float]] = None,
    lam: float = LAMBDA_DECAY, novel_weight: float = NOVEL_EDGE_WEIGHT,
) -> float:
    """TC_t = mean over transitions of [edge_weight * exp(-lam * delta_t)]."""
    if len(session_tokens) < 2:
        return 0.0

    scores = []
    for i in range(len(session_tokens) - 1):
        src, dst = session_tokens[i], session_tokens[i + 1]
        if G.has_edge(src, dst):
            w = G[src][dst]["weight"]
        else:
            w = novel_weight

        if timestamps is not None and len(timestamps) > i + 1:
            delta_t = max(timestamps[i + 1] - timestamps[i], 0.0)
        else:
            delta_t = 1.0

        scores.append(w * math.exp(-lam * delta_t))

    return float(sum(scores) / len(scores)) if scores else 0.0


def compute_tc_scores_batch(sessions_dict: Dict[str, dict], G: nx.DiGraph,
                             lam: float = LAMBDA_DECAY) -> Dict[str, float]:
    return {sid: compute_tc_score(s["tokens"], G, lam=lam) for sid, s in sessions_dict.items()}


def compute_graph_centrality(G: nx.DiGraph) -> Dict[str, float]:
    if G.number_of_edges() == 0:
        return {node: 0.0 for node in G.nodes()}
    return nx.betweenness_centrality(G, weight="weight")


def compute_gw_batch(sessions_dict: Dict[str, dict], centrality: Dict[str, float]) -> Dict[str, float]:
    scores = {}
    for sid, s in sessions_dict.items():
        tokens = s["tokens"]
        if not tokens:
            scores[sid] = 0.0
            continue
        vals = [centrality.get(t, 0.0) for t in tokens]
        scores[sid] = float(sum(vals) / len(vals))
    return scores
