# 🤖 AI Evaluation Harness

A modular and scalable framework to evaluate the performance of Large Language Models (LLMs) using structured datasets and multiple evaluation metrics.

---

## 📌 Overview

The **AI Evaluation Harness** is designed to systematically evaluate AI/LLM responses against ground truth data. It helps in measuring model accuracy, response quality, and reliability using quantitative metrics.

This project is useful for:

* AI/ML Engineers
* Data Scientists
* Developers working with LLM APIs (OpenAI, Groq, etc.)

---

## 🚀 Features

* ✅ Evaluate model responses against ground truth
* 📊 Multiple evaluation metrics (e.g., Keyword F1, Accuracy)
* 🔌 Easy integration with LLM APIs (Groq, OpenAI, etc.)
* 📁 JSON-based dataset input
* ⚡ Lightweight and easy to extend
* 📈 Clear evaluation outputs for analysis

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:**

  * pandas
  * numpy
  * scikit-learn
  * requests / API clients
* **Tools:** Git, GitHub

---

## 📂 Project Structure

```
ai-eval-harness/
│
├── data/                  # Input datasets (questions + ground truth)
├── src/                   # Core evaluation logic
├── evaluation/            # Metric calculations
├── main.py                # Entry point
├── requirements.txt       # Dependencies
├── .gitignore
├── .env.example           # Sample environment variables
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```
git clone https://github.com/DkBattulaS/Artificial-Intelligence-eval-harness.git
cd Artificial-Intelligence-eval-harness
```

---

### 2️⃣ Create virtual environment

```
python -m venv .venv
.venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Setup environment variables

Create a `.env` file and add:

```
GROQ_API_KEY=your_api_key_here
```

⚠️ Do NOT upload `.env` to GitHub.

---

### 5️⃣ Run the project

```
python main.py
```

---

## 📊 Example Input (Dataset)

```json
[
  {
    "question": "What is AI?",
    "ground_truth": "Artificial Intelligence is the simulation of human intelligence."
  }
]
```

---


## 📊 Evaluation Metrics

The system uses a **hybrid evaluation approach** combining semantic similarity, keyword-based scoring, and LLM judgment.

---

### 🔹 Metrics Used

* **Semantic Similarity**
  Measures meaning similarity using embeddings
  [
  \cos(E_A, E_G)
  ]

* **Faithfulness (Precision)**
  Checks if response stays grounded in truth
  [
  \frac{|T_{resp} \cap T_{gt}|}{|T_{resp}|}
  ]

* **Completeness (Recall)**
  Measures coverage of ground truth
  [
  \frac{|T_{resp} \cap T_{gt}|}{|T_{gt}|}
  ]

* **Hallucination Penalty**
  Penalizes irrelevant or extra content

* **LLM Evaluation**
  Judges correctness, completeness, and faithfulness

---

### 🔹 Final Score

[
0.35 \cdot \text{Semantic} +
0.25 \cdot \text{Faithfulness} +
0.15 \cdot \text{Completeness} +
0.25 \cdot \text{LLM Correctness}
]

---

### 🧠 Key Idea

A balanced combination of:

* Keyword overlap
* Semantic understanding
* LLM-based reasoning

---

* **Keyword F1 Score** – Measures overlap between predicted and actual keywords
* **Accuracy** – Checks correctness of responses
* **Custom Metrics** – Easily extendable

---

## 🔍 How It Works

1. Load dataset (questions + ground truth)
2. Send question to LLM API
3. Receive model response
4. Compare with ground truth
5. Compute evaluation metrics
6. Display results in structured format

---

## 🎯 Use Cases

* Benchmarking LLM performance
* Comparing multiple AI models
* Testing prompt engineering strategies
* Academic and research projects

---

## 🔐 Security Note

* API keys are stored in `.env`
* `.env` is excluded via `.gitignore`
* Never expose secrets in public repositories

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repo and submit a pull request.

---

## 📬 Contact

**Deepak Battula**
📧 [deepakbattula040@gmail.com](mailto:deepakbattula040@gmail.com)
🔗 GitHub: https://github.com/DkBattulaS

---

## ⭐ Acknowledgements

* Groq APIs
* Python open-source ecosystem

---

## 🌟 Show your support

If you like this project, give it a ⭐ on GitHub!
