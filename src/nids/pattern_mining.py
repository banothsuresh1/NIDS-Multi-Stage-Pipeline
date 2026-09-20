"""
Stage 8 — FP-Growth + PrefixSpan Pattern Mining.

Mines frequent *unordered* token itemsets (FP-Growth) and frequent
*ordered* token subsequences with a gap constraint (PrefixSpan-style) from
real (non-SMOTE) training sessions, then scores each held-out session by
how strongly it matches the mined attack vocabulary (SP_t).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from config import FP_GROWTH_MIN_SUPPORT, PREFIXSPAN_MAX_GAP, PREFIXSPAN_MIN_SUPPORT


def build_transaction_db(sessions_dict: Dict[str, dict]) -> Tuple["object", pd.DataFrame]:
    """One transaction per session = the set of unique tokens it contains."""
    from mlxtend.preprocessing import TransactionEncoder

    transactions = [sorted(set(s["tokens"])) for s in sessions_dict.values()]
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_transactions = pd.DataFrame(te_array, columns=te.columns_)
    return te, df_transactions


def run_fpgrowth(df_transactions: pd.DataFrame, min_support: float = FP_GROWTH_MIN_SUPPORT) -> pd.DataFrame:
    from mlxtend.frequent_patterns import fpgrowth

    if df_transactions.empty or df_transactions.shape[1] == 0:
        return pd.DataFrame(columns=["support", "itemsets"])
    fp_patterns = fpgrowth(df_transactions, min_support=min_support, use_colnames=True)
    return fp_patterns.sort_values("support", ascending=False).reset_index(drop=True)


def _subsequences_with_gap(tokens: List[str], length: int, max_gap: int):
    """Yield ordered tuples of `length` tokens where consecutive picks are within max_gap."""
    n = len(tokens)
    if length == 1:
        for t in tokens:
            yield (t,)
        return
    for i in range(n):
        for j in range(i + 1, min(i + 1 + max_gap, n)):
            yield (tokens[i], tokens[j])


def run_prefixspan(
    sessions_dict: Dict[str, dict],
    min_support: float = PREFIXSPAN_MIN_SUPPORT,
    max_gap: int = PREFIXSPAN_MAX_GAP,
) -> List[Tuple[Tuple[str, ...], float]]:
    """
    Lightweight sequential pattern miner: counts 1-token and gap-constrained
    2-token ordered patterns, keeping those above min_support.
    """
    sequences = [s["tokens"] for s in sessions_dict.values() if len(s["tokens"]) > 0]
    n_sessions = len(sequences)
    if n_sessions == 0:
        return []

    pattern_counts: Dict[Tuple[str, ...], int] = {}
    for tokens in sequences:
        seen_patterns = set()
        for pattern in _subsequences_with_gap(tokens, 1, max_gap):
            seen_patterns.add(pattern)
        for pattern in _subsequences_with_gap(tokens, 2, max_gap):
            seen_patterns.add(pattern)
        for pattern in seen_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    results = []
    for pattern, count in pattern_counts.items():
        support = count / n_sessions
        if support >= min_support:
            results.append((pattern, support))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _ordered_occurs(pattern: Tuple[str, ...], tokens: List[str], max_gap: int = PREFIXSPAN_MAX_GAP) -> bool:
    """Check whether `pattern` occurs as an ordered (gap-constrained) subsequence of tokens."""
    pos = -1
    for tok in pattern:
        found = None
        for idx in range(pos + 1, min(pos + 1 + max_gap + 1, len(tokens)) if pos >= 0 else len(tokens)):
            if tokens[idx] == tok:
                found = idx
                break
        if found is None:
            return False
        pos = found
    return True


def compute_sp_score(session_tokens: List[str], fp_patterns: pd.DataFrame,
                      ps_patterns: List[Tuple[Tuple[str, ...], float]]) -> float:
    """Mean support of all mined patterns (itemset or ordered) that match this session."""
    token_set = set(session_tokens)
    matched_supports = []

    if fp_patterns is not None and len(fp_patterns) > 0:
        for _, row in fp_patterns.iterrows():
            itemset = row["itemsets"]
            if set(itemset).issubset(token_set):
                matched_supports.append(float(row["support"]))

    for pattern, support in ps_patterns:
        if _ordered_occurs(pattern, session_tokens):
            matched_supports.append(support)

    if not matched_supports:
        return 0.0
    return float(min(sum(matched_supports) / len(matched_supports), 1.0))


def compute_sp_scores_batch(
    sessions_dict: Dict[str, dict], fp_patterns: pd.DataFrame,
    ps_patterns: List[Tuple[Tuple[str, ...], float]],
) -> Dict[str, float]:
    return {
        sid: compute_sp_score(s["tokens"], fp_patterns, ps_patterns)
        for sid, s in sessions_dict.items()
    }
