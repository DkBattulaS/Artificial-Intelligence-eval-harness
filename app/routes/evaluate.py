from fastapi import APIRouter
import json
import os
import math

from app.services.llm_service import get_llm_response
from app.services.evaluation_service import evaluate_responses

router = APIRouter()


# 🔧 Helper function to fix NaN issue
def clean_nan(data):
    return [
        {
            k: (0 if isinstance(v, float) and math.isnan(v) else v)
            for k, v in row.items()
        }
        for row in data
    ]


@router.get("/run-evaluation")
def run_evaluation():
    try:
        # ✅ Correct file path (important for stability)
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        file_path = os.path.join(BASE_DIR, "tests", "sample_testset.json")

        with open(file_path) as f:
            test_data = json.load(f)

        results = []

        for item in test_data[:3]:
            try:
                response = get_llm_response(item["question"])
            except Exception as e:
                # ✅ Handle API failure safely
                response = "Error"

            results.append({
                "question": item["question"],
                "answer": response,
                "ground_truth": item["ground_truth"]
            })

        eval_result = evaluate_responses(results)

        # ✅ Convert to safe JSON (fix NaN crash)
        return clean_nan(eval_result)

    except Exception as e:
        # ✅ Debug-friendly error output
        return {"error": str(e)}