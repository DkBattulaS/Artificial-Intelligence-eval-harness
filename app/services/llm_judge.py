import numpy as np
from sentence_transformers import util

def local_llm_judge(question, answer, ground_truth, emb_answer, emb_gt):
    # Semantic similarity
    similarity = float(util.cos_sim(emb_answer, emb_gt).item())

    # Length-based completeness
    len_ratio = min(len(answer) / (len(ground_truth) + 1), 2)

    # Heuristic scoring
    correctness = similarity
    completeness = min(1.0, similarity + 0.2)
    faithfulness = similarity * 0.8

    # Reason generation (simple but useful)
    if similarity > 0.8:
        reason = "Answer is highly relevant and matches ground truth semantically."
    elif similarity > 0.6:
        reason = "Answer is correct but more detailed than ground truth."
    else:
        reason = "Answer has low semantic alignment with ground truth."

    return {
        "correctness": round(correctness, 4),
        "completeness": round(completeness, 4),
        "faithfulness": round(faithfulness, 4),
        "reason": reason
    }