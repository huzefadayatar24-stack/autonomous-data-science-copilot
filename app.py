import os
import io
import time
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.datasets import load_diabetes, load_iris
from openai import OpenAI
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Autonomous Data Science Copilot",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------------
# SIDEBAR: CONFIGURATION & DATA INGESTION
# ---------------------------------------------------------
st.sidebar.title("⚙️ Engine & Data")
st.sidebar.success("Powered by Groq LPUs for lightning-fast execution.")

# Check if the key is securely hidden in Streamlit Secrets
if "GROQ_API_KEY" in st.secrets:
    groq_key = st.secrets["GROQ_API_KEY"]
else:
    # Fallback: Show the password box if running locally or secrets aren't set
    groq_key = st.sidebar.text_input(
        "Groq API Key", 
        type="password",
        help="Enter your Groq key (starts with gsk_)"
    )

client = None
if groq_key:
    client = OpenAI(
        api_key=groq_key, 
        base_url="https://api.groq.com/openai/v1"
    )
    model_name = "llama-3.3-70b-versatile" 

def call_llm(prompt: str, temperature: float = 0.2) -> str:
    """Helper to query Groq."""
    if not client:
        st.stop()
        raise ValueError("AI Client not initialized. Please enter your Groq API Key.")
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content

def extract_python_code(content: str) -> str:
    """Extracts raw executable code from markdown blocks."""
    if "```python" in content:
        return content.split("```python")[1].split("```")[0].strip()
    elif "```" in content:
        return content.split("```")[1].split("```")[0].strip()
    return content.strip()

# Dataset Ingestion
st.sidebar.subheader("Dataset Ingestion")
data_source = st.sidebar.radio(
    "Select Source",
    ["Iris (Classification)", "Diabetes (Regression)", "Upload CSV"]
)

df = None
target_col = None

if data_source == "Iris (Classification)":
    data = load_iris(as_frame=True)
    df = data.frame
    target_col = "target"
elif data_source == "Diabetes (Regression)":
    data = load_diabetes(as_frame=True)
    df = data.frame
    target_col = "target"
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        target_col = st.sidebar.selectbox("Select Target Column", options=df.columns)

# ---------------------------------------------------------
# MAIN INTERFACE
# ---------------------------------------------------------
st.title("🤖 Autonomous Data Science Copilot")
st.markdown("An end-to-end multi-agent pipeline for autonomous EDA, data cleaning, and machine learning optimization.")

if df is not None:
    st.subheader("📋 Dataset Preview")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.dataframe(df.head(5), use_container_width=True)
    with col2:
        st.metric("Total Rows", df.shape[0])
        st.metric("Total Columns", df.shape[1])
        st.metric("Target Variable", target_col)

    tabs = st.tabs(["📊 Dynamic EDA", "🧹 Data Cleaning", "🚀 Multi-Agent ML Lab"])

    # ---------------------------------------------------------
    # TAB 1: DYNAMIC EDA
    # ---------------------------------------------------------
    with tabs[0]:
        st.header("Exploratory Data Analysis Agent")
        if st.button("Run Dynamic EDA Agent", type="primary"):
            if not client:
                st.error("Please enter your Groq API Key in the sidebar.")
            else:
                with st.spinner("Agent analyzing dataset and generating custom plots..."):
                    dataset_summary = df.describe(include="all").to_string()
                    dtypes_info = str(df.dtypes)
                    
                    viz_prompt = f"""
You are an Expert Data Analyst.
Dataset Schema:
{dtypes_info}
Statistical Summary:
{dataset_summary}

Write Python code using matplotlib/seaborn to generate the 2 most insightful figures for this data.
Rules:
1. Data is in DataFrame named `df`.
2. Generate two figures: `fig1` and `fig2`.
3. Output ONLY Python code inside ```python ... ``` blocks.
"""
                    code = extract_python_code(call_llm(viz_prompt, temperature=0.1))
                    local_scope = {"df": df.copy(), "pd": pd, "np": np, "plt": plt, "sns": sns}
                    try:
                        exec(code, local_scope)
                        fig1 = local_scope.get("fig1")
                        fig2 = local_scope.get("fig2")
                        
                        col_a, col_b = st.columns(2)
                        if fig1: col_a.pyplot(fig1)
                        if fig2: col_b.pyplot(fig2)
                    except Exception as e:
                        st.warning(f"Dynamic plotting fallback due to code error: {e}")

                    summary_prompt = f"Data Summary:\n{dataset_summary}\nProvide a 4-bullet executive summary on distributions, variance, correlations, and preprocessing recommendations."
                    st.markdown("### 📝 Executive Summary")
                    st.markdown(call_llm(summary_prompt, temperature=0.2))

    # ---------------------------------------------------------
    # TAB 2: DATA Cleaning
    # ---------------------------------------------------------
    with tabs[1]:
        st.header("Data Cleaning Agent")
        if st.button("Run Data Cleaning Agent"):
            if not client:
                st.error("Please enter your Groq API Key in the sidebar.")
            else:
                with st.spinner("Writing transformation logic..."):
                    clean_prompt = f"Data sample:\n{df.head(5).to_dict()}\nWrite `clean_dataset(df)` to handle missing values, encode categories, and drop zero-variance columns. Return ONLY executable python code."
                    code = extract_python_code(call_llm(clean_prompt, temperature=0.1))
                    local_scope = {"df": df.copy(), "pd": pd, "np": np}
                    try:
                        exec(code, local_scope)
                        cleaned_df = local_scope["clean_dataset"](df.copy())
                        st.session_state["cleaned_df"] = cleaned_df
                        st.success("Dataset successfully cleaned!")
                        st.dataframe(cleaned_df.head(5), use_container_width=True)
                    except Exception as e:
                        st.error(f"Cleaning failed: {e}")

    # ---------------------------------------------------------
    # TAB 3: MULTI-AGENT ML LAB
    # ---------------------------------------------------------
    with tabs[2]:
        st.header("Recursive Multi-Agent Optimization")
        rounds = st.slider("Optimization Rounds", 1, 5, 3)
        
        if st.button("Start Optimization Pipeline", type="primary"):
            if not client:
                st.error("Please enter your Groq API Key in the sidebar.")
            else:
                active_df = st.session_state.get("cleaned_df", df)
                is_class = active_df[target_col].nunique() <= 10 and active_df[target_col].dtype in ['object', 'int64', 'category']
                metric_name = "Accuracy" if is_class else "R2 Score"
                
                history = []
                best_score = -float("inf")
                best_code = None

                progress_bar = st.progress(0)
                for r in range(1, rounds + 1):
                    st.markdown(f"#### 🔄 Iteration {r}/{rounds}")
                    history_txt = "\n".join([f"Round {h['round']}: {metric_name}={h['score']} | Critic: {h['feedback']}" for h in history])
                    
                    arch_prompt = f"Target: '{target_col}'. History:\n{history_txt}\nWrite `def train_and_evaluate(df):` returning `(float(score), trained_model)`. Output ONLY executable code."
                    code = extract_python_code(call_llm(arch_prompt, temperature=0.2))
                    
                    local_scope = {"df": active_df.copy(), "pd": pd, "np": np, "joblib": joblib}
                    score, error = None, None
                    try:
                        exec(code, local_scope)
                        score, model_obj = local_scope["train_and_evaluate"](active_df.copy())
                        st.write(f"✅ **Validation {metric_name}:** `{score:.4f}`")
                        if score > best_score:
                            best_score = score
                            best_code = code
                    except Exception:
                        error = traceback.format_exc()
                        st.write(f"⚠️ **Error:** `{error.splitlines()[-1]}`")

                    feedback = call_llm(f"Metric: {metric_name}, Score: {score}, Error: {error}. Critique in 2 sentences.", temperature=0.2)
                    st.info(f"**Critic Analysis:** {feedback}")
                    
                    history.append({"round": r, "score": score if score else 0.0, "feedback": feedback})
                    progress_bar.progress(r / rounds)

                st.success(f"🏆 Optimization Finished! Highest {metric_name}: {best_score:.4f}")
                if best_code:
                    with st.expander("View Winning Pipeline Code"):
                        st.code(best_code, language="python")
else:
    st.info("Select or upload a dataset using the sidebar to begin.")
