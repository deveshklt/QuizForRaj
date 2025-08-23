# quiz_firestore_app_final.py
import os
import re
import json
import uuid
import pandas as pd
import streamlit as st
import pdfplumber
from openai import OpenAI
import firebase_admin
from firebase_admin import credentials, firestore
import time

# --------------------- Firebase Initialization ---------------------
# Load Firebase credentials from Streamlit secrets
if not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase"]))
    firebase_admin.initialize_app(cred)

db = firestore.client()

# --------------------- Utilities (Updated and Combined) ---------------------
def extract_json(llm_output: str):
    """
    Tries to extract a valid JSON array from a string,
    even if the LLM response is messy.
    """
    cleaned = llm_output.strip()
    
    # Remove code fences if they exist
    cleaned = re.sub(r"^```(json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    
    # Find the JSON array
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in LLM output")
    
    json_str = match.group(0)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Simple fix for a common trailing comma issue
        fixed = re.sub(r",\s*([\]}])", r"\1", json_str)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON after fixing: {e}")

def pdf_to_txt(uploaded_pdf) -> str:
    all_text = ""
    with pdfplumber.open(uploaded_pdf) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n\n"
    return all_text

def parse_raw_blocks(text: str):
    """
    Improved function to split text into blocks.
    Handles various numbering formats and removes headers/footers.
    """
    # Remove common headers/footers
    text = re.sub(r"(P\.T\.O\.|---.*?---|.*?\n|\s*SPZ\d+\s+\d+\s*\n)", "", text, flags=re.I)
    
    # Normalize numbering to ensure a newline before each question number
    normalized_text = re.sub(r'\s*(\d+)\.\s*', r'\n\1. ', text).strip()
    
    # Split the text into individual questions based on the normalized numbers
    questions = re.split(r'\n(\d+\.\s+)', normalized_text)
    
    # Filter out empty strings and re-join question numbers with their text
    questions_list = []
    if questions and len(questions) > 1:
        for i in range(1, len(questions), 2):
            questions_list.append(questions[i] + (questions[i+1] if i+1 < len(questions) else ''))
    
    return questions_list

def llm_clean_questions_streaming(raw_questions: str):
    """
    Sends a batch of raw questions to an LLM and streams the response.
    The prompt is refined to reduce hallucinations and request null for bad data.
    """
    prompt = f"""
You are a Quiz Formatter.
You will be given messy exam questions (Hindi + English) with options. 
Clean them and output a JSON list of objects. For any content you cannot confidently extract, use `null`. Do not make up information.

The JSON format must be:
{{
 "question_english": "...",
 "options": {{"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."}}
}}

Creat only english versions if available. 
Ensure options are in the correct A/B/C/D/E order as provided.
Only return the JSON list. Do NOT use any markdown formatting, code fences, or additional text.
If a question or an option is missing or malformed, set its value to null.

Questions:
{raw_questions}
"""
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0,
        stream=True,
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        yield content

@st.cache_data
def process_quiz_batches(full_raw_text, batch_size=10, max_retries=3):
    """
    Splits the raw text into batches and processes each with retry logic.
    This function will be cached to prevent re-runs on UI interactions.
    """
    questions_list = parse_raw_blocks(full_raw_text)
    
    if len(questions_list) == 0:
        st.error("Error: No questions were found in the provided text.")
        return []

    master_list = []
    total_batches = (len(questions_list) + batch_size - 1) // batch_size
    
    progress_bar = st.progress(0, text="Processing batches...")
    status_text = st.empty()

    for i in range(0, len(questions_list), batch_size):
        batch_number = i // batch_size + 1
        retries = 0
        
        while retries < max_retries:
            try:
                batch = questions_list[i:i + batch_size]
                batch_text = "\n".join(batch)
                
                status_text.info(f"Processing batch {batch_number} of {total_batches} (Attempt {retries + 1}/{max_retries})...")
                
                full_response = ""
                for chunk in llm_clean_questions_streaming(batch_text):
                    full_response += chunk
                
                parsed_data = extract_json(full_response)
                master_list.extend(parsed_data)
                
                progress_bar.progress(batch_number / total_batches)
                break 

            except Exception as e:
                retries += 1
                status_text.warning(f"Error on batch {batch_number}: {e}. Retrying in 5s...")
                time.sleep(5)
                if retries == max_retries:
                    status_text.error(f"Failed to process batch {batch_number} after {max_retries} retries. Skipping.")
        
    progress_bar.empty()
    status_text.success("Batch processing finished.")
    return master_list

# --------------------- Streamlit Tabs ---------------------
st.set_page_config(page_title="Quiz Platform", layout="wide")
tab1, tab2 = st.tabs(["Admin Panel", "Take Quiz"])

st.set_page_config(page_title="Quiz Platform", layout="wide")
tab1, tab2 = st.tabs(["Admin Panel", "Take Quiz"])
