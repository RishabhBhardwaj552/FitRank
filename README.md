# FitRank

A cross-encoder fine-tuned to score resume-job relevance, built to evaluate whether fine-tuning on weakly-labeled data actually improves ranking quality over pretrained baselines, and by how much.

## Motivation

Most resume-job matching demos report a single accuracy number without saying where it came from or how confident it is. This project starts from the opposite question: given a cheap, imperfect labeling heuristic, how much of a ranking model's quality actually comes from fine-tuning, how does that compare to simpler baselines, and how solid is the evidence for the improvement given a small eval set.

## Approach

1. **Weak supervision.** 6,729 resume-job pairs were labeled using a keyword/category heuristic (the same logic used in JobMatch AI's retrieval layer), split across 6 job categories.
2. **Independent evaluation set.** 168 pairs across 24 held-out resumes (no overlap with training resumes) were hand-labeled independently of the heuristic. The heuristic agrees with human judgment 69% of the time, and of the pairs it calls relevant, only 39.7% actually are, which is the core motivation for fine-tuning rather than shipping the heuristic directly.
3. **Fine-tuning.** `cross-encoder/ms-marco-MiniLM-L-6-v2` fine-tuned with binary cross-entropy loss (3 epochs, batch size 16, lr 2e-5, resume text truncated to 250 words to match inference-time behavior).
4. **Baseline comparison.** The fine-tuned model is compared against three baselines of increasing sophistication: TF-IDF + cosine similarity, a pretrained bi-encoder (`all-MiniLM-L6-v2`), and the pretrained cross-encoder before fine-tuning.
5. **Statistical validation.** Resume-level bootstrap confidence intervals (2,000 resamples) and a wrong-to-right / right-to-wrong error analysis on the top-1 pick per resume.
6. **Ablation.** The same fine-tuning run repeated on training subsets of 500, 1,000, 2,000, and 4,000 pairs to check where returns on additional labeled data start to flatten.

## Results

| Tier | AUC-ROC | Precision@1 |
|---|---|---|
| TF-IDF + cosine | 0.824 | 0.667 (12/18) |
| Bi-encoder (pretrained) | 0.772 | 0.722 (13/18) |
| Cross-encoder (pretrained) | 0.739 | 0.611 (11/18) |
| Cross-encoder (fine-tuned) | **0.871** | **0.778 (14/18)** |

Precision@1 is computed over the 18 of 24 eval resumes that had at least one truly relevant candidate in their pool.

Ranking is not monotonic with architecture size: TF-IDF beats the pretrained cross-encoder on this eval set. Fine-tuning, not raw model capacity, is what produces the improvement.

**Confidence.** 95% bootstrap CI for AUC-ROC: pretrained cross-encoder [0.642, 0.839], fine-tuned [0.789, 0.940]. The intervals overlap somewhat, a reflection of the small eval set (18 resumes), but the point estimates and the error analysis both consistently favor the fine-tuned model.

**Error analysis.** Fine-tuning flipped the top-1 pick from wrong to right for 7 resumes and from right to wrong for 4, a net improvement consistent with the aggregate precision@1 gain (11/18 to 14/18).

**Ablation.** AUC-ROC improves sharply from 500 to 2,000 training pairs (+0.075) then flattens (2,000 to 6,729 pairs gains only +0.018), suggesting most of the benefit of this fine-tuning setup is captured well before the full training pool is used.

## Repository structure

```
data/            training pairs, hand-labeled eval set, ablation and baseline results
notebooks/       fine-tuning (Colab), ablation study (Colab), full analysis notebook
scripts/         cross-encoder baseline/fine-tuned comparison script
tools/           browser-based tool used to hand-label the eval set
bi_encoder_baseline.py   pretrained bi-encoder baseline
```

Raw source datasets and trained model weights are excluded from version control; the training and ablation notebooks regenerate them.

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Fine-tuning and the ablation study run on Colab (GPU); baseline scoring and the analysis notebook run locally.
