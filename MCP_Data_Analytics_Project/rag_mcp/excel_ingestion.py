"""Pandas-based Excel ingestion with worksheet and table-region detection.

Table boundaries are inferred from completely blank rows and columns. This is
heuristic: workbooks whose tables touch or rely only on formatting boundaries
may need workbook-specific rules.
"""
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

VERTICAL_MARKERS = {
    "field", "fields", "attribute", "attributes", "property", "properties",
    "metric", "parameter", "item", "particulars", "description",
}


def _occupied_bands(indices: List[int]) -> List[Tuple[int, int]]:
    if not indices:
        return []
    indices = sorted(set(indices))
    bands = []
    start = previous = indices[0]
    for value in indices[1:]:
        if value != previous + 1:
            bands.append((start, previous))
            start = value
        previous = value
    bands.append((start, previous))
    return bands


def _normalise_cell(value):
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    return value


def _header_score(row: pd.Series, following_rows: pd.DataFrame) -> float:
    values = [v for v in row.tolist() if v is not None and str(v).strip()]
    if len(values) < 2:
        return -1.0
    text_ratio = sum(isinstance(v, str) for v in values) / len(values)
    unique_ratio = len({str(v).strip().lower() for v in values}) / len(values)
    score = text_ratio * 2.0 + unique_ratio
    if not following_rows.empty:
        below_nonempty = following_rows.head(4).notna().sum(axis=1).mean()
        score += min(below_nonempty / max(len(row), 1), 1.0)
    score += min(len(values) / max(len(row), 1), 1.0)
    return score


def _find_header_row(df: pd.DataFrame) -> int:
    if len(df) < 2:
        return 0
    candidates = min(len(df) - 1, 8)
    scored = [(_header_score(df.iloc[i], df.iloc[i + 1:]), i) for i in range(candidates)]
    best_score, best_index = max(scored, key=lambda item: (item[0], -item[1]))
    return best_index if best_score >= 2.0 else 0


def _make_unique_headers(values) -> List[str]:
    """Create unique internal labels while detecting case/space-insensitive duplicates.

    Example: ["Name", " Score ", "name"] becomes
    ["Name", "Score", "name__2"]. Original labels are retained separately
    in table metadata by _prepare_table().
    """
    headers, counts = [], {}
    used = set()
    for i, value in enumerate(values):
        original = " ".join(str(value).split()) if pd.notna(value) else ""
        name = original
        if not name or name.lower().startswith("unnamed"):
            name = f"Column_{i + 1}"
        normalised = " ".join(name.split()).casefold()
        counts[normalised] = counts.get(normalised, 0) + 1
        candidate = name if counts[normalised] == 1 else f"{name}__{counts[normalised]}"
        # Avoid collisions with literal names such as "Name__2".
        suffix = counts[normalised]
        while candidate.casefold() in used:
            suffix += 1
            candidate = f"{name}__{suffix}"
        used.add(candidate.casefold())
        headers.append(candidate)
    return headers


def _duplicate_header_details(values) -> List[Dict]:
    """Return duplicate header groups using trimmed, case-insensitive matching."""
    groups = {}
    for position, value in enumerate(values, start=1):
        label = str(value).strip() if value is not None else ""
        if not label or label.lower().startswith("unnamed"):
            continue
        key = " ".join(label.split()).casefold()
        groups.setdefault(key, []).append({"position": position, "original": label})
    return [
        {"header": entries[0]["original"], "occurrences": entries}
        for entries in groups.values() if len(entries) > 1
    ]


def _normalise_orientation(df: pd.DataFrame):
    if df.shape[1] < 3 or df.shape[0] < 2:
        return df, "horizontal_or_key_value"
    first_header = str(df.columns[0]).strip().lower()
    first_values = [str(v).strip().lower() for v in df.iloc[:, 0].dropna().tolist()]
    unique_ratio = len(set(first_values)) / max(len(first_values), 1)
    if first_header in VERTICAL_MARKERS and unique_ratio > 0.75:
        transposed = df.set_index(df.columns[0]).T
        transposed.index.name = None
        transposed.columns = _make_unique_headers(list(transposed.columns))
        return transposed.reset_index(drop=True), "transposed_field_by_record"
    return df, "horizontal"


def _region_is_key_value(df: pd.DataFrame) -> bool:
    """Only normalize explicitly labelled key/value blocks, not every 2-col table."""
    if df.shape[1] != 2 or df.shape[0] < 2:
        return False
    first_header = str(df.columns[0]).strip().lower()
    second_header = str(df.columns[1]).strip().lower()
    value_headers = {"value", "values", "result", "results", "answer", "data", "measurement"}
    is_label_value_header = (
        first_header in VERTICAL_MARKERS
        or second_header in value_headers
        or second_header.endswith(" value")
    )
    if not is_label_value_header:
        return False
    labels = df.iloc[:, 0].dropna().astype(str).str.strip()
    return len(labels) >= 2 and labels.str.lower().nunique() / len(labels) >= 0.8


def _prepare_table(raw_df: pd.DataFrame, sheet_name: str, table_number: int, region: Dict):
    if raw_df.empty or raw_df.shape[0] < 2:
        return None
    header_idx = _find_header_row(raw_df)
    original_headers = raw_df.iloc[header_idx].tolist()
    duplicate_headers = _duplicate_header_details(original_headers)
    headers = _make_unique_headers(original_headers)
    body = raw_df.iloc[header_idx + 1:].copy()
    body.columns = headers
    body = body.dropna(how="all").reset_index(drop=True)
    if body.empty:
        return None

    body, orientation = _normalise_orientation(body)
    if _region_is_key_value(body):
        record = {}
        for _, row in body.iterrows():
            key, value = str(row.iloc[0]).strip(), row.iloc[1]
            if key and pd.notna(value):
                record[key] = value
        if record:
            body = pd.DataFrame([record])
            orientation = "key_value_record"

    body = body.dropna(axis=1, how="all")
    if body.empty:
        return None
    return {
        "dataframe": body,
        "metadata": {
            "sheet": sheet_name,
            "table": f"Table {table_number}",
            "region": region,
            "orientation": orientation,
            "header_row_in_region": header_idx + 1,
            "original_headers": [str(v).strip() if v is not None else "" for v in original_headers],
            "duplicate_headers": duplicate_headers,
        },
    }


def _detect_tables_in_sheet(sheet_df: pd.DataFrame, sheet_name: str) -> List[Dict]:
    """Find rectangular occupied regions separated by fully blank rows/columns."""
    if sheet_df.empty:
        return []
    sheet_df = sheet_df.copy().where(pd.notna(sheet_df), None)
    occupied_rows = [i for i in range(len(sheet_df)) if sheet_df.iloc[i].notna().any()]
    occupied_cols = [i for i in range(len(sheet_df.columns)) if sheet_df.iloc[:, i].notna().any()]
    detected = []
    table_number = 0
    for r_start, r_end in _occupied_bands(occupied_rows):
        for c_start, c_end in _occupied_bands(occupied_cols):
            raw = sheet_df.iloc[r_start:r_end + 1, c_start:c_end + 1].reset_index(drop=True)
            raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")
            if raw.empty or raw.shape[0] < 2:
                continue
            table_number += 1
            item = _prepare_table(raw, sheet_name, table_number, {
                "start_row": r_start + 1,
                "end_row": r_end + 1,
                "start_column": c_start + 1,
                "end_column": c_end + 1,
            })
            if item:
                detected.append(item)
    return detected


def _detect_project_plan(sheet_df, sheet_name):
    """Recognize project sections from labels and task headers, independent of formatting."""
    project = None
    sections = []
    for row_number, row in enumerate(sheet_df.itertuples(index=False, name=None)):
        labels = [" ".join(str(value).split()) if pd.notna(value) else "" for value in row]
        if "Project" in labels:
            index = labels.index("Project")
            project = next(
                (label.lstrip(": ").strip() for label in labels[index + 1:] if label),
                None,
            )
            sections.append((row_number, None, project))
        if {"#", "Module", "Status", "Completion %"}.issubset(labels):
            if "Task" in labels or "WBS Task" in labels:
                sections.append((row_number, labels.index("#"), project))

    tables = []
    for index, (header_row, id_col, project) in enumerate(sections):
        if id_col is None:
            continue
        end_row = sections[index + 1][0] if index + 1 < len(sections) else len(sheet_df)
        header = sheet_df.iloc[header_row]
        columns = [column for column, label in enumerate(header) if pd.notna(label)]
        body = sheet_df.iloc[header_row + 1:end_row]
        task_ids = pd.to_numeric(body.iloc[:, id_col], errors="coerce")
        body = body.loc[task_ids.notna()].iloc[:, columns].copy().reset_index(drop=True)
        body.columns = _make_unique_headers(header.iloc[columns])
        for name in ("Completion %", "Target Manday", "Overdue Count", "Duration"):
            if name in body:
                body[name] = pd.to_numeric(body[name], errors="coerce")
        tables.append({
            "dataframe": body,
            "metadata": {
                "sheet": sheet_name,
                "table": f"Table {len(tables) + 1}",
                "project": project,
                "orientation": "horizontal",
            },
        })
    return tables


def load_excel_workbook(file_path: str) -> Dict:
    """Read every worksheet into Pandas and detect tables in each worksheet."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix == ".xls":
        engine = "xlrd"
    elif suffix == ".xlsx":
        engine = "openpyxl"
    else:
        raise ValueError("Only .xls and .xlsx workbooks are supported.")

    # sheet_name=None is important: it loads all worksheets, not only the first.
    workbook_data = pd.read_excel(path, sheet_name=None, header=None, engine=engine)
    sheet_names = list(workbook_data.keys())
    tables = []
    for sheet_name, sheet_df in workbook_data.items():
        project_tables = _detect_project_plan(sheet_df, sheet_name) if sheet_name == "DevPlan" else []
        tables.extend(project_tables or _detect_tables_in_sheet(sheet_df, sheet_name))
    return {"sheet_names": sheet_names, "tables": tables}


def load_excel_tables(file_path: str) -> List[Dict]:
    """Backward-compatible helper returning only detected tables."""
    return load_excel_workbook(file_path)["tables"]
