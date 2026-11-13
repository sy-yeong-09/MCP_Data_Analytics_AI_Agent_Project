"""Full-workbook Pandas analysis helpers used alongside semantic RAG retrieval.

This module performs safe, deterministic calculations directly on the ingested
DataFrames. It does not execute LLM-generated Python code.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import pandas as pd


AGGREGATIONS = {
    "sum": ("sum", r"\b(total|sum|summation|overall)\b"),
    "mean": ("mean", r"\b(average|avg|mean)\b"),
    "max": ("max", r"\b(maximum|max|highest|largest|most|top)\b"),
    "min": ("min", r"\b(minimum|min|lowest|smallest|least)\b"),
    "count": ("count", r"\b(count|how many|number of)\b"),
}


def _normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _numeric_series(series: pd.Series) -> Optional[pd.Series]:
    """Return a numeric series only when most non-empty values are numeric."""
    non_empty = series.dropna()
    if non_empty.empty:
        return None
    converted = pd.to_numeric(non_empty.astype(str).str.replace(",", "", regex=False)
                              .str.replace(r"^\$", "", regex=True), errors="coerce")
    if converted.notna().mean() < 0.70:
        return None
    return converted.dropna()


def _column_match(question: str, columns: List[str]) -> Optional[str]:
    """Choose the most plausible column explicitly mentioned in the question."""
    q = _normalise(question)
    if not q:
        return None
    ranked: List[Tuple[float, str]] = []
    q_tokens = set(q.split())
    stop = {"what", "is", "the", "of", "for", "in", "on", "a", "an", "by",
            "total", "sum", "average", "avg", "mean", "maximum", "minimum",
            "highest", "lowest", "largest", "smallest", "count", "how", "many",
            "overall", "please", "show", "calculate", "find", "get"}
    useful_tokens = q_tokens - stop
    for column in columns:
        name = _normalise(column)
        if not name:
            continue
        score = 0.0
        if name in q:
            score = 1.0 + len(name) / 100
        else:
            tokens = set(name.split())
            overlap = len(tokens & useful_tokens) / max(len(tokens), 1)
            if overlap:
                score = overlap * 0.8 + SequenceMatcher(None, name, q).ratio() * 0.2
        if score:
            ranked.append((score, column))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    # Require a meaningful explicit match to avoid calculating on a guessed field.
    return ranked[0][1] if ranked[0][0] >= 0.45 else None


def _detect_aggregation(question: str) -> Optional[str]:
    q = question.lower()
    for operation, (_, pattern) in AGGREGATIONS.items():
        if re.search(pattern, q):
            return operation
    return None


def _detect_group_column(question: str, columns: List[str]) -> Optional[str]:
    match = re.search(r"\b(?:by|per|for each|in each|grouped by)\s+([a-zA-Z0-9 _-]+?)(?:\?|$|\b(?:and|with)\b)", question, re.I)
    if not match:
        return None
    phrase = _normalise(match.group(1))
    candidates = []
    for column in columns:
        name = _normalise(column)
        if name == phrase or name in phrase or phrase in name:
            candidates.append(column)
    if candidates:
        return max(candidates, key=lambda c: len(_normalise(c)))
    return _column_match(phrase, columns)



def _detect_winner_column(question: str, columns: List[str]) -> Optional[str]:
    """Find an entity column in questions such as 'which product has highest sales'."""
    match = re.search(r"\bwhich\s+([a-zA-Z0-9 _-]+?)\s+(?:has|have|with|generated|recorded|made)\b", question, re.I)
    if not match:
        return None
    phrase = _normalise(match.group(1))
    for column in columns:
        name = _normalise(column)
        if name == phrase or name in phrase or phrase in name:
            return column
    return _column_match(phrase, columns)


def _format_number(value: object) -> str:
    try:
        number = float(value)
        if number.is_integer():
            return f"{int(number):,}"
        return f"{number:,.4f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(value)


def _aggregate(values: pd.Series, operation: str):
    if operation == "sum":
        return values.sum()
    if operation == "mean":
        return values.mean()
    if operation == "max":
        return values.max()
    if operation == "min":
        return values.min()
    if operation == "count":
        return int(values.count())
    raise ValueError(f"Unsupported aggregation: {operation}")


def _workbook_profile(tables: List[Dict], max_columns: int = 18) -> str:
    """Compact descriptive profile from all tables, not just retrieved chunks."""
    lines = ["FULL-WORKBOOK DESCRIPTIVE PROFILE (computed with Pandas):"]
    included = 0
    for item in tables:
        df = item["dataframe"].copy()
        meta = item["metadata"]
        if df.empty:
            continue
        lines.append(f"\nSheet '{meta['sheet']}', {meta['table']}: {len(df)} data rows.")
        numeric_found = 0
        for column in list(df.columns)[:max_columns]:
            series = _numeric_series(df[column])
            if series is not None:
                lines.append(
                    f"- {column}: count={series.count()}, mean={_format_number(series.mean())}, "
                    f"median={_format_number(series.median())}, min={_format_number(series.min())}, "
                    f"max={_format_number(series.max())}, sum={_format_number(series.sum())}."
                )
                numeric_found += 1
            elif df[column].notna().any():
                top = df[column].dropna().astype(str).value_counts().head(3)
                if not top.empty:
                    examples = ", ".join(f"{k} ({v})" for k, v in top.items())
                    lines.append(f"- {column}: {df[column].nunique(dropna=True)} distinct values; common values: {examples}.")
            included += 1
        if numeric_found == 0:
            lines.append("- No clearly numeric columns detected in this table.")
    if len(lines) == 1:
        return "No tabular data was available for descriptive profiling."
    return "\n".join(lines)


def analyze_workbook(tables: List[Dict], question: str) -> Dict:
    """Calculate explicit numerical requests against full DataFrames.

    Returns handled=False when the question is not a clearly identifiable
    aggregate request; callers can then use retrieved RAG context and profile.
    """
    profile = _workbook_profile(tables)
    operation = _detect_aggregation(question)
    if operation is None:
        return {"handled": False, "kind": "descriptive_or_semantic", "text": profile}

    all_columns = []
    numeric_columns = []
    for item in tables:
        frame = item["dataframe"]
        all_columns.extend(str(c) for c in frame.columns)
        for col in frame.columns:
            if operation == "count" or _numeric_series(frame[col]) is not None:
                numeric_columns.append(str(col))
    unique_columns = list(dict.fromkeys(all_columns))
    candidate_columns = list(dict.fromkeys(numeric_columns)) if operation != "count" else unique_columns
    column = _column_match(question, candidate_columns)
    if column is None:
        return {"handled": False, "kind": "needs_column_clarification", "text": profile}

    group_column = _detect_group_column(question, unique_columns)
    winner_column = _detect_winner_column(question, unique_columns) if operation in {"max", "min"} else None
    pieces = []
    matching_rows = 0
    numeric_parts = []
    grouped_frames = []
    winner_frames = []
    for item in tables:
        df = item["dataframe"].copy()
        meta = item["metadata"]
        matching = next((c for c in df.columns if str(c) == column), None)
        if matching is None:
            continue
        if operation == "count":
            values = df[matching].dropna()
        else:
            values = _numeric_series(df[matching])
            if values is None:
                continue
        matching_rows += len(values)
        numeric_parts.append(values)
        if winner_column and winner_column in df.columns:
            winner_temp = df[[winner_column, matching]].copy()
            winner_temp[matching] = pd.to_numeric(
                winner_temp[matching].astype(str).str.replace(",", "", regex=False)
                .str.replace(r"^\$", "", regex=True), errors="coerce")
            winner_temp = winner_temp.dropna(subset=[winner_column, matching])
            if not winner_temp.empty:
                winner_frames.append(winner_temp)
        if group_column and group_column in df.columns:
            temp = df[[group_column, matching]].copy()
            if operation != "count":
                temp[matching] = pd.to_numeric(
                    temp[matching].astype(str).str.replace(",", "", regex=False)
                    .str.replace(r"^\$", "", regex=True), errors="coerce")
            temp = temp.dropna(subset=[group_column, matching])
            if not temp.empty:
                grouped_frames.append(temp)
        pieces.append(f"- Sheet '{meta['sheet']}', {meta['table']}: {len(values)} valid values.")

    if not numeric_parts:
        return {"handled": False, "kind": "needs_numeric_column", "text": profile}

    combined = pd.concat(numeric_parts, ignore_index=True)
    label = {"sum": "Total", "mean": "Average", "max": "Maximum", "min": "Minimum", "count": "Count"}[operation]
    result_lines = [
        "PANDAS FULL-WORKBOOK CALCULATION:",
        f"Question interpreted as: {label.lower()} of '{column}'.",
        f"Number of valid values included: {matching_rows:,}.",
    ]

    if winner_column and winner_frames and not group_column:
        winner_data = pd.concat(winner_frames, ignore_index=True)
        winner_row = winner_data.loc[winner_data[column].idxmax() if operation == "max" else winner_data[column].idxmin()]
        direction = "highest" if operation == "max" else "lowest"
        result_lines.append(
            f"{winner_column} with {direction} {column}: {winner_row[winner_column]} "
            f"({_format_number(winner_row[column])})."
        )
    elif group_column and grouped_frames:
        grouped_data = pd.concat(grouped_frames, ignore_index=True)
        grouped_series = grouped_data.groupby(group_column)[column]
        if operation == "count":
            grouped_all = grouped_series.count()
        elif operation == "sum":
            grouped_all = grouped_series.sum()
        elif operation == "mean":
            grouped_all = grouped_series.mean()
        elif operation == "max":
            grouped_all = grouped_series.max()
        else:
            grouped_all = grouped_series.min()
        result_lines.append(f"Grouped by '{group_column}' ({label.lower()}):")
        for key, value in grouped_all.sort_values(ascending=(operation == "min")).items():
            result_lines.append(f"- {key}: {_format_number(value)}")
    else:
        result = _aggregate(combined, operation)
        result_lines.append(f"{label} {column}: {_format_number(result)}")
        if operation in {"sum", "mean", "count"}:
            result_lines.append(
                f"Additional statistics: min={_format_number(combined.min())}, "
                f"max={_format_number(combined.max())}, "
                f"mean={_format_number(combined.mean())}, median={_format_number(combined.median())}."
            )
    result_lines.extend(pieces)
    return {
        "handled": True,
        "kind": "pandas_numerical",
        "text": "\n".join(result_lines),
        "profile": profile,
        "column": column,
        "operation": operation,
        "group_column": group_column,
    }
