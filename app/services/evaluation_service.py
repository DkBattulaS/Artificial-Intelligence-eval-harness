from sentence_transformers import SentenceTransformer, util
import re
import numpy as np
from app.services.llm_judge import local_llm_judge

# ✅ Load model once 
model = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Better stopwords 
STOPWORDS = {
    "is", "a", "the", "of", "and", "to", "in", "that", "it", "an", "are",
    "for", "on", "with", "as", "by", "this", "was", "be", "or"
}

# ✅ Text preprocessing 
def normalize(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

# ✅ Extract meaningful keywords
def key_terms(text):
    words = re.findall(r'\b\w+\b', normalize(text))
    return set(w for w in words if w not in STOPWORDS and len(w) > 2)

# ✅ Batch embedding 
def get_embeddings(texts):
    return model.encode(texts, convert_to_tensor=True)

# ✅ Semantic similarity
def semantic_score(emb1, emb2):
    return round(float(util.cos_sim(emb1, emb2).item()), 4)

# ✅ Precision (Faithfulness)
def faithfulness_score(resp_terms, gt_terms):
    if not resp_terms:
        return 0.0
    overlap = resp_terms & gt_terms
    return len(overlap) / len(resp_terms)

# ✅ Recall (Completeness)
def completeness_score(resp_terms, gt_terms):
    if not gt_terms:
        return 1.0
    overlap = resp_terms & gt_terms
    return len(overlap) / len(gt_terms)

# ✅ Hallucination penalty 
def hallucination_penalty(resp_terms, gt_terms):
    if not resp_terms:
        return 0.0

    overlap = len(resp_terms & gt_terms)
    ratio = overlap / (len(resp_terms) + 1)

    # dynamic penalty
    penalty = 1 - ratio

    # clamp values
    return round(max(0.1, min(0.6, penalty)), 4)

def compute_scores(answer, ground_truth, question, emb_answer, emb_gt):
    resp_terms = key_terms(answer)
    gt_terms = key_terms(ground_truth)

    precision = faithfulness_score(resp_terms, gt_terms)
    recall = completeness_score(resp_terms, gt_terms)
    penalty = hallucination_penalty(resp_terms, gt_terms)
    semantic = semantic_score(emb_answer, emb_gt)

    # ✅ Improved faithfulness
    faithfulness = (
        0.5 * semantic +
        0.2 * precision +
        0.3 * (1 - penalty)
    )

   
    llm_score = local_llm_judge(
        question,
        answer,
        ground_truth,
        emb_answer,
        emb_gt
    )

    # ✅ Final Score
    final_score = (
        0.35 * semantic +
        0.25 * faithfulness +
        0.15 * recall +
        0.25 * llm_score["correctness"]
    )

    return {
        "answer_relevancy": round(semantic, 4),
        "faithfulness": round(faithfulness, 4),
        "completeness": round(recall, 4),

        # LLM scores
        "llm_correctness": llm_score["correctness"],
        "llm_completeness": llm_score["completeness"],
        "llm_faithfulness": llm_score["faithfulness"],
        "llm_reason": llm_score["reason"],

        "hallucination_penalty": round(penalty, 4),
        "final_score": round(final_score, 4)
    }
# ✅ Main evaluation pipeline
def evaluate_responses(data):
    results = []

    # 🚀 Batch embeddings (VERY IMPORTANT optimization)
    answers = [normalize(item["answer"]) for item in data]
    gts = [normalize(item["ground_truth"]) for item in data]

    answer_embeddings = get_embeddings(answers)
    gt_embeddings = get_embeddings(gts)

    for i, item in enumerate(data):
        scores = compute_scores(
            answers[i],
            gts[i],
            item["question"],
            answer_embeddings[i],
            gt_embeddings[i]
        )

        results.append({
            "question": item["question"],
            "answer": item["answer"],
            "ground_truth": item["ground_truth"],
            "user_input": item["question"],
            "response": item["answer"],
            "reference": item["ground_truth"],
            **scores
        })

    return results