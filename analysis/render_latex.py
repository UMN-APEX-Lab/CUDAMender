"""Render pandas DataFrames as booktabs LaTeX tables with optional row-max bolding."""
from __future__ import annotations

from typing import Literal

import pandas as pd

BoldMode = Literal["none", "row_max"]


def _escape(s: str) -> str:
    """Escape LaTeX special chars in a string."""
    return (
        s.replace("\\", "\\textbackslash ")
         .replace("&", "\\&")
         .replace("%", "\\%")
         .replace("$", "\\$")
         .replace("#", "\\#")
         .replace("_", "\\_")
         .replace("{", "\\{")
         .replace("}", "\\}")
    )


def _fmt_value(v) -> str:
    if v is None or (isinstance(v, float) and (v != v)):  # NaN
        return "--"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def render_table(
    df: pd.DataFrame,
    index_col: str,
    bold: BoldMode = "none",
    caption: str | None = None,
    footnote: str | None = None,
) -> str:
    """Render df as a LaTeX booktabs tabular.

    Args:
        df: must contain `index_col`; remaining columns are data.
        bold: "row_max" → bold the largest numeric value per row across data cols;
              "none" → no bolding (use for multi-metric tables with mixed units).
        caption: optional table caption (wrapped in \\caption).
        footnote: optional small text after the table (wrapped in \\footnotesize).
    """
    if index_col not in df.columns:
        raise ValueError(f"index_col {index_col!r} not in df.columns: {list(df.columns)}")
    data_cols = [c for c in df.columns if c != index_col]
    n_data = len(data_cols)

    lines: list[str] = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    if caption:
        lines.append(f"\\caption{{{_escape(caption)}}}")
    lines.append("\\begin{tabular}{l" + "r" * n_data + "}")
    lines.append("\\toprule")
    header = [_escape(index_col)] + [_escape(c) for c in data_cols]
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")

    for _, row in df.iterrows():
        bold_idx = None
        if bold == "row_max":
            numeric_vals = []
            for c in data_cols:
                v = row[c]
                if v is not None and not (isinstance(v, float) and v != v):
                    numeric_vals.append((v, c))
            if numeric_vals:
                _, bold_col = max(numeric_vals)
                bold_idx = data_cols.index(bold_col)

        cells = [_escape(str(row[index_col]))]
        for i, c in enumerate(data_cols):
            text = _fmt_value(row[c])
            if i == bold_idx and text != "--":
                text = f"\\textbf{{{text}}}"
            cells.append(text)
        lines.append(" & ".join(cells) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if footnote:
        lines.append(f"\\par\\footnotesize{{{_escape(footnote)}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)
