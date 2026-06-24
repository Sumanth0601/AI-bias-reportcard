from datetime import datetime

STATUS_COLOR = {"PASS": "#22c55e", "WARN": "#f59e0b", "FAIL": "#ef4444"}
STATUS_BG = {"PASS": "#dcfce7", "WARN": "#fef3c7", "FAIL": "#fee2e2"}

METRIC_EXPLANATIONS = {
    "demographic_parity_difference": "Measures whether the AI predicts positive outcomes at similar rates across groups. Values above 0.1 suggest potential unfairness.",
    "equalized_odds_difference": "Measures whether the AI correctly identifies positive cases at equal rates across groups (equal recall). Values above 0.1 suggest the model misses more cases in some groups.",
    "disparate_impact_ratio": "The ratio of the lowest to highest positive prediction rate across groups. Below 0.8 triggers the legal 'four-fifths rule' threshold for potential discrimination.",
}

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f1f5f9; color: #1e293b; padding: 24px; }
.container { max-width: 960px; margin: 0 auto; }
.card { background: #fff; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; }
.subtitle { color: #64748b; font-size: 0.9rem; margin-bottom: 20px; }
.verdict { border-radius: 8px; padding: 16px 24px; font-size: 1.2rem; font-weight: 700; margin-bottom: 24px; color: #fff; }
.section-header { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
.section-header h2 { font-size: 1.1rem; font-weight: 600; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 99px; font-size: 0.78rem; font-weight: 700; color: #fff; }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-bottom: 16px; }
th { background: #f8fafc; text-align: left; padding: 8px 12px; border-bottom: 2px solid #e2e8f0; font-weight: 600; }
td { padding: 8px 12px; border-bottom: 1px solid #f1f5f9; }
.low-cell { background: #fee2e2; }
.disparity-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
.disparity-item { border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px 16px; }
.disparity-label { font-size: 0.78rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 4px; }
.disparity-value { font-size: 1.4rem; font-weight: 700; margin-bottom: 4px; }
.disparity-explain { font-size: 0.78rem; color: #64748b; }
footer { text-align: center; font-size: 0.8rem; color: #94a3b8; margin-top: 8px; }
footer a { color: #6366f1; text-decoration: none; }
@media print { .no-print { display: none; } body { background: #fff; padding: 0; } .card { box-shadow: none; } }
"""


def _badge(status: str) -> str:
    color = STATUS_COLOR.get(status, "#94a3b8")
    return f'<span class="badge" style="background:{color}">{status}</span>'


def _render_group_table(groups: dict) -> str:
    metrics = ["accuracy", "recall", "fpr", "fnr"]
    # Find min value index per metric (lower is worse for accuracy/recall, higher is worse for fpr/fnr)
    worst = {}
    for m in ["accuracy", "recall"]:
        vals = {g: groups[g][m] for g in groups if groups[g][m] is not None}
        if vals:
            worst[m] = min(vals, key=vals.get)
    for m in ["fpr", "fnr"]:
        vals = {g: groups[g][m] for g in groups if groups[g][m] is not None}
        if vals:
            worst[m] = max(vals, key=vals.get)

    rows = ""
    for g, data in groups.items():
        def cell(metric, val):
            cls = ' class="low-cell"' if worst.get(metric) == g else ""
            display = f"{val:.1%}" if val is not None else "N/A"
            return f"<td{cls}>{display}</td>"

        rows += f"""<tr>
            <td><strong>{g}</strong></td>
            <td>{data['size']}</td>
            {cell('accuracy', data['accuracy'])}
            {cell('recall', data['recall'])}
            {cell('fpr', data['fpr'])}
            {cell('fnr', data['fnr'])}
        </tr>"""

    return f"""<table>
        <thead><tr>
            <th>Group</th><th>Size</th><th>Accuracy</th><th>Recall</th>
            <th>False Positive Rate</th><th>False Negative Rate</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>"""


def _render_disparity_panel(disparity: dict) -> str:
    items = [
        ("demographic_parity_difference", "Demographic Parity Difference", disparity["demographic_parity_difference"], disparity["dpd_status"]),
        ("equalized_odds_difference", "Equalized Odds Difference", disparity["equalized_odds_difference"], disparity["eod_status"]),
        ("disparate_impact_ratio", "Disparate Impact Ratio", disparity["disparate_impact_ratio"], disparity["dir_status"]),
    ]

    cards = ""
    for key, label, value, status in items:
        color = STATUS_COLOR.get(status, "#94a3b8")
        val_str = f"{value:.4f}" if value is not None else "N/A"
        cards += f"""<div class="disparity-item">
            <div class="disparity-label">{label}</div>
            <div class="disparity-value" style="color:{color}">{val_str} {_badge(status)}</div>
            <div class="disparity-explain">{METRIC_EXPLANATIONS[key]}</div>
        </div>"""

    return f'<div class="disparity-grid">{cards}</div>'


def render_report(metrics_dict: dict, filename: str) -> str:
    all_statuses = [v["disparity"]["overall_status"] for v in metrics_dict.values()]

    if "FAIL" in all_statuses:
        verdict_text = "BIAS DETECTED"
        verdict_color = STATUS_COLOR["FAIL"]
    elif "WARN" in all_statuses:
        verdict_text = "REVIEW REQUIRED"
        verdict_color = STATUS_COLOR["WARN"]
    else:
        verdict_text = "PASS"
        verdict_color = STATUS_COLOR["PASS"]

    sections = ""
    for col, data in metrics_dict.items():
        status = data["disparity"]["overall_status"]
        sections += f"""<div class="card">
            <div class="section-header">
                <h2>{col}</h2>
                {_badge(status)}
            </div>
            {_render_group_table(data['groups'])}
            {_render_disparity_panel(data['disparity'])}
        </div>"""

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Fairness Report Card</title>
<style>{CSS}</style>
</head>
<body>
<div class="container">
    <div class="card">
        <h1>AI Fairness Report Card</h1>
        <div class="subtitle">{filename} &mdash; Generated {timestamp}</div>
        <div class="verdict" style="background:{verdict_color}">{verdict_text}</div>
    </div>
    {sections}
    <footer class="card">
        <p>Generated by <a href="https://github.com/sumanthratna/AI-bias-reportcard">AI Bias Report Card</a></p>
        <p style="margin-top:6px">This report is for informational purposes. Consult a qualified AI ethics expert before making compliance determinations.</p>
    </footer>
</div>
</body>
</html>"""
