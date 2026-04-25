import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="talk")


def read_results_text(file_path: Path) -> str:
    raw = file_path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_model_and_language(name: str) -> tuple[str, str]:
    """Return canonical model id and language code from result filename-like keys."""
    patterns = [
        # e.g. qwen.en.json, finetuned_qwen_0.5B_en.json
        r"^(?P<model>.+?)[._](?P<lang>en|es)\.json$",
        # e.g. gemini_en_predictions.json
        r"^(?P<model>.+?)[._](?P<lang>en|es)_predictions\.json$",
    ]

    for pattern in patterns:
        match = re.match(pattern, name, flags=re.IGNORECASE)
        if match:
            model = match.group("model")
            lang = match.group("lang").upper()
            return model, lang

    return name, "UNK"


def clean_label(model_name: str) -> str:
    label = model_name
    label = label.replace("finetuned_", "FT ")
    label = label.replace("_", " ")
    return label


def model_family(model_name: str) -> str:
    lower = model_name.lower()
    if lower.startswith("finetuned_qwen"):
        return "Fine-tuned Qwen"
    if lower.startswith("finetuned_gemma"):
        return "Fine-tuned Gemma"
    if lower.startswith("qwen"):
        return "Qwen"
    if lower.startswith("ministral"):
        return "Ministral"
    if lower.startswith("gemma"):
        return "Gemma"
    if lower.startswith("gemini"):
        return "Gemini"
    return "Other"


def parse_results(file_path: Path) -> pd.DataFrame:
    text = read_results_text(file_path)
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        json_start = line.find("{")
        if json_start < 0:
            continue

        name = line[:json_start].strip()
        if name.endswith(":"):
            name = name[:-1].strip()

        metrics_part = line[json_start:].strip()
        try:
            metrics = json.loads(metrics_part)
        except json.JSONDecodeError:
            continue

        if "f1" not in metrics or "exact_match" not in metrics:
            continue

        base_name, lang = parse_model_and_language(name)

        rows.append(
            {
                "file_name": name,
                "model": base_name,
                "label": clean_label(base_name),
                "family": model_family(base_name),
                "lang": lang,
                "f1": float(metrics["f1"]),
                "exact_match": float(metrics["exact_match"]),
                "is_finetuned": base_name.lower().startswith("finetuned_"),
            }
        )

    if not rows:
        raise ValueError("No parseable results were found in results.txt")

    df = pd.DataFrame(rows)
    df["lang"] = pd.Categorical(df["lang"], categories=["EN", "ES", "UNK"], ordered=True)
    return df


def save_fig(fig, out_path: Path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def chart_grouped_metric(df: pd.DataFrame, metric: str, title: str, out_path: Path):
    pivot = (
        df.pivot_table(index="label", columns="lang", values=metric, aggfunc="mean")
        .dropna(subset=["EN", "ES"], how="any")
        .assign(avg=lambda x: x.mean(axis=1))
        .sort_values("avg", ascending=False)
    )

    labels = pivot.index.tolist()
    en_vals = pivot["EN"].to_numpy()
    es_vals = pivot["ES"].to_numpy()

    x = np.arange(len(labels))
    width = 0.42

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.65), 7))
    bars_en = ax.bar(x - width / 2, en_vals, width, label="EN", color="#1f77b4")
    bars_es = ax.bar(x + width / 2, es_vals, width, label="ES", color="#d62728")

    ax.set_title(title, pad=14)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.legend()

    ax.bar_label(bars_en, fmt="%.1f", padding=2, fontsize=8)
    ax.bar_label(bars_es, fmt="%.1f", padding=2, fontsize=8)

    save_fig(fig, out_path)


def chart_language_gap(df: pd.DataFrame, out_path: Path):
    pivot = df.pivot_table(index="label", columns="lang", values="f1", aggfunc="mean").dropna(subset=["EN", "ES"]) 
    gap = (pivot["EN"] - pivot["ES"]).sort_values(ascending=False)

    colors = ["#2ca02c" if g >= 0 else "#9467bd" for g in gap]
    fig, ax = plt.subplots(figsize=(11, max(6, len(gap) * 0.38)))
    ax.barh(gap.index, gap.values, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Language Gap by Model (EN F1 - ES F1)")
    ax.set_xlabel("F1 Gap (points)")
    ax.set_ylabel("Model")

    for i, val in enumerate(gap.values):
        ax.text(val + (0.2 if val >= 0 else -0.2), i, f"{val:.1f}", va="center", ha="left" if val >= 0 else "right", fontsize=9)

    save_fig(fig, out_path)


def chart_metrics_heatmap(df: pd.DataFrame, out_path: Path):
    pivot_f1 = df.pivot_table(index="label", columns="lang", values="f1", aggfunc="mean")
    pivot_em = df.pivot_table(index="label", columns="lang", values="exact_match", aggfunc="mean")

    table = pd.DataFrame(index=pivot_f1.index)
    table["EN_F1"] = pivot_f1.get("EN")
    table["ES_F1"] = pivot_f1.get("ES")
    table["EN_EM"] = pivot_em.get("EN")
    table["ES_EM"] = pivot_em.get("ES")
    table["AVG_F1"] = table[["EN_F1", "ES_F1"]].mean(axis=1)
    table = table.dropna(subset=["EN_F1", "ES_F1", "EN_EM", "ES_EM"], how="any")
    table = table.sort_values("AVG_F1", ascending=False)

    fig, ax = plt.subplots(figsize=(9, max(6, len(table) * 0.45)))
    sns.heatmap(table.round(2), annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.5, cbar_kws={"label": "Score"}, ax=ax)
    ax.set_title("Model Score Heatmap (F1 and Exact Match)")
    save_fig(fig, out_path)


def chart_family_summary(df: pd.DataFrame, out_path: Path):
    df = df[df["lang"].isin(["EN", "ES"])].copy()
    df["lang"] = df["lang"].astype(str)
    summary = (
        df.groupby(["family", "lang"], as_index=False)[["f1", "exact_match"]]
        .mean()
        .sort_values(["f1", "exact_match"], ascending=False)
    )

    families = summary["family"].drop_duplicates().tolist()

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(families) * 0.5)), sharey=True)
    sns.barplot(data=summary, x="f1", y="family", hue="lang", palette=["#1f77b4", "#d62728"], ax=axes[0])
    axes[0].set_title("Average F1 by Model Family")
    axes[0].set_xlabel("F1")
    axes[0].set_ylabel("Family")

    sns.barplot(data=summary, x="exact_match", y="family", hue="lang", palette=["#1f77b4", "#d62728"], ax=axes[1])
    axes[1].set_title("Average Exact Match by Model Family")
    axes[1].set_xlabel("Exact Match")
    axes[1].set_ylabel("")

    for ax in axes:
        ax.set_xlim(0, 100)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title="Language")

    save_fig(fig, out_path)


def normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def find_base_for_finetuned(ft_model: str, base_models: list[str]) -> str | None:
    ft_norm = normalize_token(ft_model.replace("finetuned_", "", 1))
    ft_norm_variants = {
        ft_norm,
        ft_norm.replace("it", ""),
        ft_norm.replace("pt", ""),
        ft_norm.replace("base", ""),
    }

    if ft_norm.startswith("qwen"):
        ft_norm_variants.add(ft_norm.replace("qwen", "qwen15"))

    family = ""
    for candidate in ("qwen", "gemma", "ministral", "gemini"):
        if ft_norm.startswith(candidate):
            family = candidate
            break

    for base_model in base_models:
        base_norm = normalize_token(base_model)
        if family and not base_norm.startswith(family):
            continue

        for variant in ft_norm_variants:
            if not variant:
                continue
            if variant == base_norm or variant in base_norm or base_norm in variant:
                return base_model

    return None


def chart_finetune_gains(df: pd.DataFrame, out_path: Path):
    df = df[df["lang"].isin(["EN", "ES"])].copy()
    df["lang"] = df["lang"].astype(str)
    base_models = [m for m in df["model"].unique() if not m.lower().startswith("finetuned_")]
    rows = []

    finetuned_models = [m for m in df["model"].unique() if m.lower().startswith("finetuned_")]
    for ft_model in finetuned_models:
        base_model = find_base_for_finetuned(ft_model, base_models)
        if base_model is None:
            continue

        for lang in ("EN", "ES"):
            base_row = df[(df["model"] == base_model) & (df["lang"] == lang)]
            ft_row = df[(df["model"] == ft_model) & (df["lang"] == lang)]
            if base_row.empty or ft_row.empty:
                continue

            rows.append(
                {
                    "pair": f"{clean_label(base_model)} -> {clean_label(ft_model)}",
                    "lang": lang,
                    "f1_gain": float(ft_row["f1"].iloc[0] - base_row["f1"].iloc[0]),
                    "em_gain": float(ft_row["exact_match"].iloc[0] - base_row["exact_match"].iloc[0]),
                }
            )

    gains = pd.DataFrame(rows)
    if gains.empty:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(gains["pair"].unique()) * 1.4)), sharey=True)
    sns.barplot(data=gains, x="f1_gain", y="pair", hue="lang", palette=["#1f77b4", "#d62728"], ax=axes[0])
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_title("Fine-Tuning Gains: F1")
    axes[0].set_xlabel("F1 Improvement")
    axes[0].set_ylabel("Base -> Fine-Tuned")

    sns.barplot(data=gains, x="em_gain", y="pair", hue="lang", palette=["#1f77b4", "#d62728"], ax=axes[1])
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_title("Fine-Tuning Gains: Exact Match")
    axes[1].set_xlabel("Exact Match Improvement")
    axes[1].set_ylabel("")

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title="Language")

    save_fig(fig, out_path)
    return True


def write_summary(df: pd.DataFrame, out_path: Path):
    pivot = df.pivot_table(index="label", columns="lang", values="f1", aggfunc="mean")
    pivot = pivot.dropna(subset=["EN", "ES"], how="any")
    pivot["AVG_F1"] = pivot[["EN", "ES"]].mean(axis=1)
    pivot["GAP_EN_MINUS_ES"] = pivot["EN"] - pivot["ES"]

    top_en = df[df["lang"] == "EN"].sort_values("f1", ascending=False).iloc[0]
    top_es = df[df["lang"] == "ES"].sort_values("f1", ascending=False).iloc[0]
    smallest_gap = pivot.reindex(pivot["GAP_EN_MINUS_ES"].abs().sort_values().index).iloc[0]
    smallest_gap_model = pivot.reindex(pivot["GAP_EN_MINUS_ES"].abs().sort_values().index).index[0]

    lines = [
        "Summary of model performance",
        "",
        f"Best EN F1: {top_en['label']} ({top_en['f1']:.2f})",
        f"Best ES F1: {top_es['label']} ({top_es['f1']:.2f})",
        f"Smallest EN-ES gap: {smallest_gap_model} ({smallest_gap['GAP_EN_MINUS_ES']:.2f})",
        "",
        "Top 5 models by average F1:",
    ]

    top5 = pivot.sort_values("AVG_F1", ascending=False).head(5)
    for model_name, row in top5.iterrows():
        lines.append(f"- {model_name}: EN={row['EN']:.2f}, ES={row['ES']:.2f}, AVG={row['AVG_F1']:.2f}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive visualizations from results.txt")
    parser.add_argument("--input", default="../results.txt", help="Path to results file")
    # parser.add_argument("--output-dir", default="comprehensive_visualizations", help="Directory to save charts")
    parser.add_argument("--output-dir", default=".", help="Directory to save charts")

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = parse_results(input_path)
    df.to_csv(out_dir / "parsed_results_table.csv", index=False)

    chart_grouped_metric(
        df,
        metric="f1",
        title="All Models: English vs Spanish F1",
        out_path=out_dir / "all_models_f1_comparison.png",
    )
    chart_grouped_metric(
        df,
        metric="exact_match",
        title="All Models: English vs Spanish Exact Match",
        out_path=out_dir / "all_models_em_comparison.png",
    )
    chart_language_gap(df, out_dir / "language_gap_f1_sorted.png")
    chart_metrics_heatmap(df, out_dir / "metrics_heatmap.png")
    chart_family_summary(df, out_dir / "family_summary.png")

    has_gain_chart = chart_finetune_gains(df, out_dir / "fine_tuning_gains.png")
    write_summary(df, out_dir / "summary.txt")

    print(f"Saved visualizations to: {out_dir.resolve()}")
    if has_gain_chart:
        print("Included fine-tuning gain chart.")
    else:
        print("Skipped fine-tuning gain chart (no base/fine-tuned pairs found).")


if __name__ == "__main__":
    main()
