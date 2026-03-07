import json
import base64
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

from Config import (
    CKPT_DIR, PLOTS_DIR, UNIFIED_CLASSES,
    NUM_CLASSES, MODELS_TO_TRAIN
)

EXPLANATIONS_DIR = Path("outputs/explanations")
REPORT_DIR       = Path("outputs/report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH      = REPORT_DIR / "fer_xai_report.html"

BASELINE = {
    "model":     "AlexNet (Baseline)",
    "test_acc":  0.6500,
    "macro_f1":  0.6100,
    "mean_auc":  0.7200,
}

def img_to_base64(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = path.suffix.lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{data}"

def load_model_results(model_name: str, dataset_name: str) -> dict:
    results_path = CKPT_DIR / f"{model_name}_{dataset_name}_results.json"

    if not results_path.exists():
        print(f"  ⚠ No results found for {model_name} on {dataset_name}")
        return None

    with open(results_path) as f:
        results = json.load(f)

    print(f"  ✅ Loaded results: {model_name} | "
          f"acc={results.get('test_acc', 0):.1%}")
    return results

def load_explanation_images(
    method:       str,   # "lime" or "shap"
    model_name:   str,
    dataset_name: str,
    max_images:   int = 6
) -> list:
    folder = (EXPLANATIONS_DIR / method /
              f"{model_name}_{dataset_name}")

    if not folder.exists():
        print(f"  ⚠ No {method} explanations found for {model_name}")
        return []

    images = []
    png_files = sorted(folder.glob("*.png"))[:max_images]

    for png_path in png_files:
        b64 = img_to_base64(png_path)
        if b64:
            images.append({
                "name":  png_path.stem,
                "b64":   b64,
            })

    print(f"  ✅ Loaded {len(images)} {method} images for {model_name}")
    return images

def load_training_plots(model_name: str, dataset_name: str) -> dict:
    plots = {}

    plot_types = [
        "training_curves",
        "confusion_matrix",
        "roc_curves",
    ]

    for plot_type in plot_types:
        path = PLOTS_DIR / f"{model_name}_{dataset_name}_{plot_type}.png"
        b64  = img_to_base64(path)
        if b64:
            plots[plot_type] = b64
            print(f"  ✅ Loaded plot: {plot_type} for {model_name}")
        else:
            print(f"  ⚠ Missing plot: {plot_type} for {model_name}")

    return plots

def collect_all_data(dataset_name: str) -> dict:
    print("\n  Collecting data for report...")
    data = {
        "dataset":   dataset_name,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "baseline":  BASELINE,
        "models":    []
    }

    for model_name in MODELS_TO_TRAIN:
        print(f"\n  Processing {model_name}...")
        entry = {
            "name":    model_name,
            "results": load_model_results(model_name, dataset_name),
            "plots":   load_training_plots(model_name, dataset_name),
            "lime":    load_explanation_images(
                           "lime", model_name, dataset_name),
            "shap":    load_explanation_images(
                           "shap", model_name, dataset_name),
        }
        data["models"].append(entry)

    return data

def html_header(dataset_name: str, generated: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>XAI-FER Report — {dataset_name}</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg:       #f8fafc;
            --surface:  #ffffff;
            --border:   #e2e8f0;
            --accent:   #6366f1;
            --accent2:  #06b6d4;
            --text:     #0f172a;
            --muted:    #64748b;
            --success:  #10b981;
            --warning:  #f59e0b;
            --danger:   #ef4444;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: var(--bg);
            color: var(--text);
        }}

        /* ── Hero ── */
        .hero {{
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
            padding: 60px 40px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }}

        .hero::before {{
            content: '';
            position: absolute;
            width: 600px; height: 600px;
            background: radial-gradient(circle, rgba(99,102,241,0.2) 0%, transparent 70%);
            top: -200px; left: 50%;
            transform: translateX(-50%);
            pointer-events: none;
        }}

        .hero-pill {{
            display: inline-block;
            background: rgba(99,102,241,0.2);
            border: 1px solid rgba(99,102,241,0.4);
            color: #a5b4fc;
            font-size: 0.75em;
            font-weight: 500;
            padding: 5px 16px;
            border-radius: 100px;
            letter-spacing: 2px;
            text-transform: uppercase;
            margin-bottom: 20px;
        }}

        .hero h1 {{
            font-size: clamp(1.8em, 4vw, 3em);
            font-weight: 700;
            color: #fff;
            line-height: 1.2;
            margin-bottom: 12px;
        }}

        .hero h1 span {{
            background: linear-gradient(90deg, #818cf8, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .hero-sub {{
            color: #94a3b8;
            font-size: 0.95em;
            font-weight: 300;
            margin-top: 8px;
        }}

        .hero-badges {{
            display: flex;
            gap: 10px;
            justify-content: center;
            margin-top: 24px;
            flex-wrap: wrap;
        }}

        .hero-badges span {{
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255,255,255,0.12);
            color: #cbd5e1;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75em;
            padding: 5px 14px;
            border-radius: 6px;
        }}

        /* ── Container ── */
        .container {{
            max-width: 1300px;
            margin: 0 auto;
            padding: 36px 24px;
        }}

        /* ── Section ── */
        .section {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 32px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }}

        .section h2 {{
            font-size: 1.15em;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 22px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .section h2::after {{
            content: '';
            flex: 1;
            height: 1px;
            background: var(--border);
        }}

        .section h3 {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78em;
            font-weight: 500;
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin: 24px 0 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border);
        }}

        /* ── Table ── */
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9em;
        }}

        th {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.73em;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: var(--muted);
            padding: 10px 14px;
            text-align: left;
            border-bottom: 2px solid var(--border);
            font-weight: 500;
        }}

        td {{
            padding: 12px 14px;
            border-bottom: 1px solid #f1f5f9;
            color: var(--text);
        }}

        tr:last-child td {{ border-bottom: none; }}
        tr:hover td {{ background: #f8fafc; }}

        .baseline td {{ color: var(--muted); }}
        .best {{ color: var(--accent); font-weight: 600; }}

        /* ── Badges ── */
        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 3px 10px;
            border-radius: 100px;
            font-size: 0.75em;
            font-weight: 600;
            letter-spacing: 0.3px;
        }}

        .badge-green {{
            background: #d1fae5;
            color: #065f46;
        }}

        .badge-red {{
            background: #fee2e2;
            color: #991b1b;
        }}

        .badge-blue {{
            background: #e0e7ff;
            color: #3730a3;
        }}

        /* ── TOC ── */
        .toc {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}

        .toc a {{
            font-size: 0.85em;
            font-weight: 500;
            color: var(--accent);
            border: 1px solid #e0e7ff;
            background: #f5f3ff;
            padding: 6px 16px;
            border-radius: 8px;
            text-decoration: none;
            transition: all 0.15s;
        }}

        .toc a:hover {{
            background: var(--accent);
            border-color: var(--accent);
            color: white;
        }}

        /* ── Image grids ── */
        .img-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
            gap: 14px;
            margin-top: 14px;
        }}

        .img-card {{
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            transition: box-shadow 0.2s;
        }}

        .img-card:hover {{
            box-shadow: 0 4px 16px rgba(99,102,241,0.12);
            border-color: #c7d2fe;
        }}

        .img-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}

        .img-card .caption {{
            padding: 8px 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72em;
            color: var(--muted);
            background: #f8fafc;
            border-top: 1px solid var(--border);
        }}

        /* ── Plot row ── */
        .plot-row {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
            gap: 14px;
            margin-top: 14px;
        }}

        .plot-card {{
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            transition: box-shadow 0.2s;
        }}

        .plot-card:hover {{
            box-shadow: 0 4px 16px rgba(99,102,241,0.12);
            border-color: #c7d2fe;
        }}

        .plot-card img {{ width: 100%; height: auto; display: block; }}

        .plot-card .caption {{
            padding: 8px 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72em;
            color: var(--muted);
            background: #f8fafc;
            border-top: 1px solid var(--border);
            text-align: center;
        }}

        /* ── Footer ── */
        footer {{
            text-align: center;
            padding: 32px;
            font-size: 0.82em;
            color: var(--muted);
            border-top: 1px solid var(--border);
            margin-top: 20px;
        }}
    </style>
</head>
<body>

<div class="hero">
    <div class="hero-pill">Research Report</div>
    <h1>Explainable <span>Facial Emotion</span><br>Recognition</h1>
    <p class="hero-sub">CNN-based emotion classification with LIME & SHAP interpretability</p>
    <div class="hero-badges">
        <span>Dataset: {dataset_name.upper()}</span>
        <span>ResNet-50 · EfficientNet-B0 · VGG-16</span>
        <span>Generated: {generated}</span>
    </div>
</div>

<div class="container">
"""

def html_toc() -> str:
    return """
<div class="section">
    <h2>📋 Table of Contents</h2>
    <div class="toc">
        <a href="#comparison">Model Comparison</a>
        <a href="#training">Training Curves</a>
        <a href="#confusion">Confusion Matrices</a>
        <a href="#roc">ROC Curves</a>
        <a href="#lime">LIME Explanations</a>
        <a href="#shap">SHAP Explanations</a>
    </div>
</div>
"""

def html_footer() -> str:
    return """
    <footer>
        XAI Facial Emotion Recognition &nbsp;·&nbsp;
        ResNet-50 · EfficientNet-B0 · VGG-16 &nbsp;·&nbsp;
        LIME + SHAP Explanations
    </footer>
</div>
</body>
</html>"""

def html_comparison_table(data: dict) -> str:
    rows = ""

    # Baseline row
    b = data["baseline"]
    rows += f"""
    <tr class="baseline">
        <td>{b['model']}</td>
        <td>—</td>
        <td>{b['test_acc']:.1%}</td>
        <td>{b['macro_f1']:.1%}</td>
        <td>{b['mean_auc']:.1%}</td>
        <td><span class="badge badge-blue">Baseline</span></td>
    </tr>"""

    # Collect best values for highlighting
    accs, f1s, aucs = [], [], []
    for m in data["models"]:
        r = m["results"]
        if r:
            accs.append(r.get("test_acc", 0))
            f1s.append(r.get("macro_f1", 0))
            aucs.append(r.get("mean_auc", 0))

    best_acc = max(accs) if accs else 0
    best_f1  = max(f1s)  if f1s  else 0
    best_auc = max(aucs) if aucs else 0

    for m in data["models"]:
        r = m["results"]
        if not r:
            rows += f"""
    <tr>
        <td>{m['name']}</td>
        <td>—</td>
        <td colspan="3"><em>Not trained yet</em></td>
        <td><span class="badge badge-red">Pending</span></td>
    </tr>"""
            continue

        acc     = r.get("test_acc", 0)
        f1      = r.get("macro_f1", 0)
        auc     = r.get("mean_auc", 0)
        epochs  = r.get("best_epoch", "—")
        beats   = acc > data["baseline"]["test_acc"]

        acc_cls = "best" if acc == best_acc else ""
        f1_cls  = "best" if f1  == best_f1  else ""
        auc_cls = "best" if auc == best_auc else ""
        badge   = ('<span class="badge badge-green">Beats Baseline</span>'
                   if beats else
                   '<span class="badge badge-red">Below Baseline</span>')

        rows += f"""
    <tr>
        <td><strong>{m['name']}</strong></td>
        <td>{epochs}</td>
        <td class="{acc_cls}">{acc:.1%}</td>
        <td class="{f1_cls}">{f1:.1%}</td>
        <td class="{auc_cls}">{auc:.1%}</td>
        <td>{badge}</td>
    </tr>"""

    return f"""
<div class="section" id="comparison">
    <h2>📊 Model Comparison</h2>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Best Epoch</th>
                <th>Test Accuracy</th>
                <th>Macro F1</th>
                <th>Mean AUC</th>
                <th>vs Baseline</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
</div>"""


def html_plots_section(
    section_id: str,
    title:      str,
    plot_key:   str,
    data:       dict
) -> str:
    cards = ""
    for m in data["models"]:
        b64 = m["plots"].get(plot_key)
        if b64:
            cards += f"""
        <div class="plot-card">
            <img src="{b64}" alt="{m['name']} {plot_key}">
            <div class="caption">{m['name']}</div>
        </div>"""

    if not cards:
        cards = "<p><em>No plots available yet.</em></p>"

    return f"""
<div class="section" id="{section_id}">
    <h2>{title}</h2>
    <div class="plot-row">{cards}</div>
</div>"""


def html_explanations_section(
    section_id: str,
    title:      str,
    method:     str,
    data:       dict
) -> str:
    content = ""
    for m in data["models"]:
        images = m[method]
        if not images:
            continue
        content += f"<h3>{m['name']}</h3><div class='img-grid'>"
        for img in images:
            content += f"""
        <div class="img-card">
            <img src="{img['b64']}" alt="{img['name']}">
            <div class="caption">{img['name'].replace('_', ' ')}</div>
        </div>"""
        content += "</div>"

    if not content:
        content = "<p><em>No explanations available yet.</em></p>"

    return f"""
<div class="section" id="{section_id}">
    <h2>{title}</h2>
    {content}
</div>"""

def generate_report(dataset_name: str = "fer2013") -> Path:
    print(f"\n{'='*60}")
    print(f"  Generating XAI-FER Report")
    print(f"{'='*60}")

    # Collect all data
    data = collect_all_data(dataset_name)

    # Build HTML
    print("\n  Building HTML...")
    html = ""
    html += html_header(dataset_name, data["generated"])
    html += html_toc()
    html += html_comparison_table(data)
    html += html_plots_section(
                "training", "📈 <span class='section-num'>01</span> Training Curves",
                "training_curves", data)
    html += html_plots_section(
                "confusion", "🔢 <span class='section-num'>02</span> Confusion Matrices",
                "confusion_matrix", data)
    html += html_plots_section(
                "roc", "📉 <span class='section-num'>03</span> ROC Curves",
                "roc_curves", data)
    html += html_explanations_section(
                "lime", "🟡 <span class='section-num'>04</span> LIME Explanations",
                "lime", data)
    html += html_explanations_section(
                "shap", "🔴 <span class='section-num'>05</span> SHAP Explanations",
                "shap", data)
    html += html_footer()

    # Save report
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = REPORT_PATH.stat().st_size / 1024
    print(f"\n  ✅ Report saved → {REPORT_PATH}")
    print(f"  📦 File size: {size_kb:.1f} KB")
    print(f"\n  Open in browser:")
    print(f"  {REPORT_PATH.resolve()}")

    return REPORT_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fer2013")
    args = parser.parse_args()

    generate_report(args.dataset)