# 🤖 Autonomous Data Science Copilot

A recursive multi-agent AI system with an interactive Streamlit UI designed to automate Exploratory Data Analysis, automated data preprocessing, and iterative machine learning pipeline optimization.

---

## ⚡ Key Highlights
* **Dynamic EDA Agent:** Analyzes dataset schemas and automatically codes custom Seaborn/Matplotlib visual figures tailored to the ingested columns.
* **Autonomous Data Cleaning:** Writes and validates execution logic to handle categorical encoding, median imputation, and schema anomalies.
* **3-Agent ML Optimization Loop:**
  * **Architect:** Formulates hypotheses and writes Scikit-Learn training pipelines.
  * **Executor:** Runs generated pipelines in a sandboxed scope and computes cross-validated metrics.
  * **Critic:** Evaluates metric convergence, diagnoses bottlenecks, and guides the Architect in successive iterations.
* **Artifact Generation:** Exports cleaned datasets, standalone pipeline scripts, and serialized `best_model.pkl` binaries.

---

## 🛠️ Tech Stack
* **Language & UI:** Python, Streamlit
* **Data & Modeling:** Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib, Joblib
* **Inference Engine:** Groq API / OpenAI API Client (`llama-3.3-70b-versatile`)

---

## 🚀 Local Setup & Execution
```bash
git clone [https://github.com/](https://github.com/)<your-username>/autonomous-data-science-copilot.git
cd autonomous-data-science-copilot
pip install -r requirements.txt
streamlit run app.py
