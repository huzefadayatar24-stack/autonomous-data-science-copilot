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
    # Updated to the current active Groq model
    model_name = "llama-3.3-70b-versatile" 

def call_llm(prompt: str, temperature: float = 0.2) -> str:
    """Helper to query Groq with error handling."""
    if not client:
        st.error("AI Client not initialized. Please enter your Groq API Key in the sidebar or Secrets.")
        st.stop()
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        # This will print the exact Groq error cleanly on the screen
        st.error(f"Groq API Error: {e}")
        st.stop()

def extract_python_code(content: str) -> str:
    """Extracts raw executable code from markdown blocks."""
    if "```python" in content:
        return content.split("```python")[1].split("```")[0].strip()
    elif "```" in content:
        return content.split("
