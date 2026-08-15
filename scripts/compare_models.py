"""
Project 2 -- Evaluation: baseline vs fine-tuned cross-encoder.

Compares the original pretrained `cross-encoder/ms-marco-MiniLM-L-6-v2`
against the domain-fine-tuned version, scored on the hand-labeled eval set
(`labeled_eval_fixed.csv`), which was labeled independently of the
weak-supervision heuristic used for training -- so this comparison is not
circular.

Two metrics, both threshold-free / ranking-based (avoids picking an
arbitrary score cutoff, which differs between the two models anyway):

1. AUC-ROC: overall ability to separate relevant from not-relevant pairs
   across the whole eval set.
2. Per-resume Precision@1: for each held-out resume's ~7 candidate jobs,
   does the model's #1-ranked candidate match a human-labeled "relevant"
   job? This directly mirrors how the cross-encoder is actually used in
   production (src/reranking/cross_encoder.py's rerank(), which returns
   the top-ranked candidates for one query at a time) -- resumes with zero
   human-labeled-relevant candidates are excluded from this metric since
   "top pick is correct" isn't well-defined for them.

Usage:
    python compare_models.py \\
        --labeled_csv labeled_eval_fixed.csv \\
        --blind_csv eval_pairs_blind.csv \\
        --finetuned_dir jobmatch-cross-encoder-finetuned
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sentence_transformers.cross_encoder import CrossEncoder

BASELINE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def truncate_words(text, max_words=250):
    return " ".join(str(text).split()[:max_words])


def precision_at_1(df, score_col):
    """Per-resume: is the top-scored candidate human-labeled relevant?
    Averaged only over resumes that have >=1 relevant candidate."""
    correct, total = 0, 0
    for resume_id, group in df.groupby("resume_id"):
        if group["human_label"].sum() == 0:
            continue  # no relevant candidate exists for this resume -- skip
        top = group.loc[group[score_col].idxmax()]
        correct += int(top["human_label"] == 1)
        total += 1
    return correct / total if total else float("nan"), total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled_csv", default="labeled_eval_fixed.csv")
    parser.add_argument("--blind_csv", default="eval_pairs_blind.csv")
    parser.add_argument("--finetuned_dir", default="jobmatch-cross-encoder-finetuned")
    args = parser.parse_args()

    print("Loading eval data...")
    labeled = pd.read_csv(args.labeled_csv)
    blind = pd.read_csv(args.blind_csv)[["resume_id", "job_id", "resume_text", "job_text"]]
    df = labeled.merge(blind, on=["resume_id", "job_id"], how="left")

    missing = df["resume_text"].isna().sum()
    if missing:
        print(f"WARNING: {missing} rows couldn't be matched back to eval_pairs_blind.csv -- check file paths/versions.")
        df = df.dropna(subset=["resume_text", "job_text"])

    df["resume_text_trunc"] = df["resume_text"].apply(truncate_words)
    pairs = list(zip(df["resume_text_trunc"], df["job_text"]))
    print(f"{len(df)} labeled pairs ready  ({df['human_label'].sum()} relevant / {(df['human_label']==0).sum()} not relevant)")

    print(f"\nLoading baseline model: {BASELINE_MODEL}")
    baseline = CrossEncoder(BASELINE_MODEL, max_length=512)
    df["baseline_score"] = baseline.predict(pairs)

    print(f"Loading fine-tuned model: {args.finetuned_dir}")
    finetuned = CrossEncoder(args.finetuned_dir, max_length=512)
    df["finetuned_score"] = finetuned.predict(pairs)

    print("\n=== Results ===\n")

    for label, col in [("Baseline (pretrained)", "baseline_score"), ("Fine-tuned", "finetuned_score")]:
        auc = roc_auc_score(df["human_label"], df[col])
        p1, n_resumes = precision_at_1(df, col)
        print(f"{label:<24} AUC-ROC: {auc:.3f}   Precision@1: {p1:.3f}  (over {n_resumes} resumes w/ >=1 relevant candidate)")

    print("\nAUC-ROC: overall ability to separate relevant from not-relevant across all pairs (0.5 = random, 1.0 = perfect).")
    print("Precision@1: how often the model's top-ranked candidate per resume is actually relevant -- the production use case.")

    df.to_csv("eval_scored_comparison.csv", index=False)
    print("\nFull scored comparison saved to eval_scored_comparison.csv")


if __name__ == "__main__":
    main()
