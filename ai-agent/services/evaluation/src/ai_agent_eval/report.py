"""Write evaluation reports: machine-readable JSON (for CI/trends) + a human-readable HTML summary."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .harness import EvalResult


def write_reports(result: EvalResult, out_dir: str) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{result.suite}_report.json"
    html_path = out / f"{result.suite}_report.html"
    json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    html_path.write_text(_render_html(result), encoding="utf-8")
    return {"json": str(json_path), "html": str(html_path)}


def _render_html(result: EvalResult) -> str:
    status = "PASS" if result.passed else "FAIL"
    color = "#16a34a" if result.passed else "#dc2626"

    metrics_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{value:.3f}</td></tr>"
        for name, value in sorted(result.metrics.items())
    )
    gate_rows = (
        "".join(f"<li>{html.escape(f)}</li>" for f in result.gate_failures) or "<li>none</li>"
    )

    item_rows = ""
    for row in result.per_item:
        scores = " · ".join(f"{k}={v:.2f}" for k, v in sorted(row["scores"].items()))
        item_rows += (
            f"<tr><td>{html.escape(str(row.get('id')))}</td><td>{html.escape(scores)}</td></tr>"
        )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Eval: {html.escape(result.suite)}</title>
<style>
 body{{font-family:system-ui,sans-serif;margin:24px;color:#0f172a}}
 .status{{display:inline-block;padding:4px 12px;border-radius:999px;color:white;background:{color}}}
 table{{border-collapse:collapse;margin:12px 0}} td,th{{border:1px solid #e5e7eb;padding:6px 12px;text-align:left}}
 h2{{margin-top:24px}}
</style></head><body>
<h1>Evaluation — {html.escape(result.suite)} <span class="status">{status}</span></h1>
<p>Items: {result.total} · Judge: {html.escape(result.judge)}</p>
<h2>Aggregate metrics</h2>
<table><tr><th>metric</th><th>score</th></tr>{metrics_rows}</table>
<h2>Gate failures</h2>
<ul>{gate_rows}</ul>
<h2>Per-item</h2>
<table><tr><th>id</th><th>scores</th></tr>{item_rows}</table>
</body></html>"""
