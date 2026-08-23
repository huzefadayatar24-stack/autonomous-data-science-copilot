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

# Groq API Configuration
groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
    help="Enter your Groq API key (starts with gsk_)"
)

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
)

# Initialize OpenAI Client pointing to Groq
client = None
if groq_api_key:
    client = OpenAI(
        api_key=groq_api_key,
        base_url="https://api.groq.com/openai/v1"
    )

def call_llm(prompt: str, temperature: float = 0.2) -> str:
    """Helper to query Groq via OpenAI client."""
    if not client:
        raise ValueError("Please provide a valid Groq API Key in the sidebar.")
    response = client.chat.completions.create(
        model=model_choice,
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
        st.write("The analyst agent inspects column schemas and automatically generates custom visualizations.")
        
        if st.button("Run Dynamic EDA Agent", type="primary"):
            if not groq_api_key:
                st.error("Please enter a Groq API Key in the sidebar.")
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

Task: Write Python code using matplotlib/seaborn to generate the 2 most insightful figures for this data.
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
                        if fig1:
                            col_a.pyplot(fig1)
                        if fig2:
                            col_b.pyplot(fig2)
                        st.success("Visualizations generated successfully.")
                    except Exception as e:
                        st.warning(f"Dynamic plotting fallback: {e}")
                        # Fallback Correlation Heatmap
                        numeric_df = df.select_dtypes(include=[np.number])
                        if numeric_df.shape[1] > 1:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                            st.pyplot(fig)

                    # Executive Summary
                    summary_prompt = f"""
You are an expert Data Analyst.
Statistical Summary:
{dataset_summary}

Provide a 4-bullet executive summary on:
1. Target distribution.
2. High-variance/skewed features.
3. Key interactions/correlations.
4. Preprocessing recommendations.
"""
                    insights = call_llm(summary_prompt, temperature=0.2)
                    st.markdown("### 📝 Executive Summary")
                    st.markdown(insights)

    # ---------------------------------------------------------
    # TAB 2: DATA CLEANING
    # ---------------------------------------------------------
    with tabs[1]:
        st.header("Data Cleaning Agent")
        st.write("Generates self-executing preprocessing logic to handle missing data, encoding, and duplicates.")
        
        if st.button("Run Data Cleaning Agent"):
            if not groq_api_key:
                st.error("Please enter a Groq API Key in the sidebar.")
            else:
                with st.spinner("Data Engineer agent writing transformation logic..."):
                    clean_prompt = f"""
Given data sample:
{df.head(5).to_dict()}

Write a function `clean_dataset(df)` that handles missing values, encodes categories, and drops zero-variance columns.
Output ONLY executable code inside ```python ... ```.
"""
                    code = extract_python_code(call_llm(clean_prompt, temperature=0.1))
                    local_scope = {"df": df.copy(), "pd": pd, "np": np}
                    try:
                        exec(code, local_scope)
                        cleaned_df = local_scope["clean_dataset"](df.copy())
                        st.session_state["cleaned_df"] = cleaned_df
                        st.success("Dataset successfully cleaned and stored in memory.")
                        st.dataframe(cleaned_df.head(5), use_container_width=True)
                        
                        csv_data = cleaned_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Cleaned CSV",
                            data=csv_data,
                            file_name="cleaned_dataset.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Cleaning execution failed: {e}")

    # ---------------------------------------------------------
    # TAB 3: MULTI-AGENT ML LAB
    # ---------------------------------------------------------
    with tabs[2]:
        st.header("Recursive Multi-Agent Optimization")
        st.write("Architect, Executor, and Critic agents collaborate to formulate hypotheses, train models, and critique iterations.")
        
        rounds = st.slider("Optimization Rounds", min_value=1, max_value=5, value=3)
        
        if st.button("Start Optimization Pipeline", type="primary"):
            if not groq_api_key:
                st.error("Please enter a Groq API Key in the sidebar.")
            else:
                active_df = st.session_state.get("cleaned_df", df)
                is_classification = active_df[target_col].nunique() <= 10 and active_df[target_col].dtype in ['object', 'int64', 'category']
                metric_name = "Accuracy" if is_classification else "R2 Score"
                dataset_summary = active_df.describe().to_string()
                
                history = []
                best_score = -float("inf")
                best_pipeline = None
                best_code = None

                progress_bar = st.progress(0)
                status_container = st.container()

                for r in range(1, rounds + 1):
                    with status_container:
                        st.markdown(f"#### 🔄 Iteration {r} of {rounds}")
                        
                        # 1. Architect
                        history_text = "\n".join([
                            f"Round {h['round']}: {metric_name}={h['score']:.4f} | Feedback: {h['feedback']}"
                            for h in history
                        ]) or "No prior attempts."
                        
                        arch_prompt = f"""
You are the Lead ML Architect.
Dataset Summary:
{dataset_summary}
Target: '{target_col}' ({'Classification' if is_classification else 'Regression'}).
History:
{history_text}

Write `def train_and_evaluate(df):` returning `(float(score), trained_model)`.
Output ONLY executable code inside ```python ... ```.
"""
                        code = extract_python_code(call_llm(arch_prompt, temperature=0.2))
                        
                        # 2. Executor
                        local_scope = {"df": active_df.copy(), "pd": pd, "np": np, "joblib": joblib}
                        score, error, model_obj = None, None, None
                        try:
                            exec(code, local_scope)
                            score, model_obj = local_scope["train_and_evaluate"](active_df.copy())
                            st.write(f"✅ **Executor Validation {metric_name}:** `{score:.4f}`")
                        except Exception:
                            error = traceback.format_exc()
                            st.write(f"⚠️ **Execution Error:** `{error.splitlines()[-1]}`")

                        # 3. Critic
                        critic_prompt = f"Metric: {metric_name}, Score: {score}, Error: {error}. Critique performance in 2 sentences."
                        feedback = call_llm(critic_prompt, temperature=0.2)
                        st.info(f"**Critic Analysis:** {feedback}")

                        if score is not None and score > best_score:
                            best_score = score
                            best_pipeline = model_obj
                            best_code = code

                        history.append({"round": r, "score": score if score else 0.0, "feedback": feedback})
                        progress_bar.progress(r / rounds)
                        time.sleep(0.5)

                st.success(f"🏆 Optimization Finished! Highest {metric_name}: {best_score:.4f}")

                if best_pipeline is not None:
                    buffer = io.BytesIO()
                    joblib.dump(best_pipeline, buffer)
                    buffer.seek(0)
                    st.download_button(
                        "Download Best Model (.pkl)",
                        data=buffer,
                        file_name="best_model.pkl"
                    )

                if best_code:
                    with st.expander("View Winning Pipeline Code"):
                        st.code(best_code, language="python")
else:
    st.info("Select or upload a dataset using the sidebar to begin.")
