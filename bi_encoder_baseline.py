import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity


def precision_at_1(df, score_col, label_col='human_label'):
    correct, total = 0, 0
    for rid, grp in df.groupby('resume_id'):
        if grp[label_col].sum() == 0:
            continue
        top = grp.loc[grp[score_col].idxmax()]
        correct += int(top[label_col] == 1)
        total += 1
    return correct / total, correct, total


eval_blind = pd.read_csv('data/eval_pairs_blind.csv')
labeled = pd.read_csv('data/labeled_eval_fixed.csv')
eval_df = eval_blind.drop(columns=['human_label']).merge(
    labeled[['resume_id', 'job_id', 'human_label']], on=['resume_id', 'job_id']
)

model = SentenceTransformer('all-MiniLM-L6-v2')
resume_emb = model.encode(eval_df['resume_text'].tolist(), show_progress_bar=True)
job_emb = model.encode(eval_df['job_text'].tolist(), show_progress_bar=True)

scores = np.array([cosine_similarity([resume_emb[i]], [job_emb[i]])[0, 0] for i in range(len(eval_df))])
eval_df['bi_encoder_score'] = scores

auc = roc_auc_score(eval_df['human_label'], scores)
p1, correct, total = precision_at_1(eval_df, 'bi_encoder_score')

print(f"AUC-ROC: {auc:.3f}")
print(f"Precision@1: {p1:.3f} ({correct}/{total})")

eval_df[['resume_id', 'job_id', 'category', 'human_label', 'bi_encoder_score']].to_csv(
    'data/bi_encoder_scores.csv', index=False
)
print("saved data/bi_encoder_scores.csv")
