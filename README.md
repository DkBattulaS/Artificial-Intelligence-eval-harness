🚀 AI Evaluation Harness

An end-to-end LLM Evaluation System built with FastAPI that generates responses using a Large Language Model and evaluates them using a hybrid scoring approach.

🧠 Overview

This project evaluates AI-generated answers by comparing them with ground truth data using:

Semantic similarity (Sentence Transformers)
Keyword-based precision & recall
Hallucination penalty
Lightweight LLM-based judge

It acts as a testing framework for LLM outputs, ensuring quality, correctness, and reliability.

⚙️ Tech Stack
Backend: FastAPI
LLM: Groq (LLaMA 3)
NLP: Sentence Transformers
Language: Python

🔄 System Flow
User Request → LLM Response → Evaluation Engine → Score Output

📊 Evaluation Metrics
✅ Answer Relevancy (semantic similarity)
✅ Faithfulness (precision-based)
✅ Completeness (recall-based)
✅ Hallucination Penalty
✅ Final Weighted Score


📁 Project Structure
ai-eval-harness/
│
├── app/
│   ├── routes/
│   ├── services/
│   ├── main.py
│
├── tests/
│   └── sample_testset.json
│
├── static/
│
├── requirements.txt
├── .env.example
├── README.md


⚡ Setup Instructions
1️⃣ Clone Repository
git clone https://github.com/your-username/ai-eval-harness.git

cd ai-eval-harness
2️⃣ Install Dependencies

pip install -r requirements.txt
3️⃣ Setup Environment Variables

Create a .env file:

GROQ_API_KEY=your_api_key_here 
4️⃣ Run the Server
uvicorn app.main:app --reload
📡 API Endpoint
GET /run-evaluation


🧪 Sample Dataset

The system uses a structured dataset containing:
  Questions
  Ground truth answers
  Expected keywords

🔥 Key Features
    Hybrid evaluation (semantic + symbolic + heuristic)
    Batch embedding optimization
    Modular architecture (routes, services)
    Error handling & NaN-safe output
    Ready for dashboard integration


🚀 Future Improvements
GPT-based evaluation (LLM-as-a-judge)
React dashboard for visualization
Database for tracking evaluation history
Multi-model comparison (OpenAI, Groq, etc.)


💡 Use Cases
Evaluate chatbot responses
Benchmark LLM performance
AI quality assurance systems
Research & experimentation

👨‍💻 Author
Deepak Battula

⭐ If you found this useful, give it a star!