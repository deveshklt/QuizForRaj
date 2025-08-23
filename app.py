# --------------------- Imports ---------------------
import os, re, json, uuid, time
import pandas as pd
import streamlit as st
import pdfplumber
from openai import OpenAI
import firebase_admin
from firebase_admin import credentials, firestore

# --------------------- Firebase ---------------------
if not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase"]))
    firebase_admin.initialize_app(cred)
db = firestore.client()

# --------------------- PDF Utilities ---------------------
def pdf_to_txt(uploaded_pdf) -> str:
    all_text = []
    with pdfplumber.open(uploaded_pdf) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)
    return "\n\n".join(all_text)

def split_question_and_options(question_block: str):
    options = {"A": None, "B": None, "C": None, "D": None, "E": None}
    parts = re.split(r'([A-E][\).])', question_block)
    q_text = parts[0].strip()
    for i in range(1, len(parts)-1, 2):
        label = parts[i][0]
        option_text = parts[i+1].strip()
        options[label] = option_text if option_text else None
    return q_text, options

def parse_raw_blocks_structured(text: str):
    text = re.sub(r"(P\.T\.O\.|---.*?---|.*?\n|\s*SPZ\d+\s+\d+\s*\n|\f)", "", text, flags=re.I)
    normalized_text = re.sub(r'\s*(\d+)\.\s*', r'\n\1. ', text).strip()
    raw_questions = [q.strip() for q in re.split(r'(?:\n\d+\.\s+)', normalized_text) if q.strip()]
    structured_questions = []
    for q in raw_questions:
        q_text, options = split_question_and_options(q)
        structured_questions.append({"question": q_text, "options": options})
    return structured_questions

# --------------------- LLM Utilities ---------------------
def extract_json(llm_output: str):
    cleaned = re.sub(r"^```(json)?\s*|\s*```$", "", llm_output.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in LLM output")
    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        fixed = re.sub(r",\s*([\]}])", r"\1", json_str)
        return json.loads(fixed)

def llm_clean_questions_streaming(batch_json: str):
    prompt = f"""
You are a Quiz Formatter.
You will be given structured quiz questions with options (some may be missing or messy). 
Return only **English parts** and fill missing options with null.

Return a JSON list like:
{{
 "question_english": "...",
 "options": {{"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."}}
}}

Batch:
{batch_json}
"""
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0,
        stream=True,
    )
    for chunk in stream:
        yield chunk.choices[0].delta.content or ""

# --------------------- Batch Processing ---------------------
@st.cache_data
def process_quiz_batches_structured(full_raw_text, batch_size=10, max_retries=3):
    questions_list = parse_raw_blocks_structured(full_raw_text)
    if not questions_list:
        st.error("Error: No questions found in the PDF.")
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
                batch_payload = [{"question": q["question"], "options": q["options"]} for q in batch]
                batch_text = json.dumps(batch_payload, ensure_ascii=False, indent=2)

                status_text.info(f"Processing batch {batch_number}/{total_batches} (Attempt {retries+1})...")
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
                    status_text.error(f"Failed batch {batch_number}. Skipping.")
    
    progress_bar.empty()
    status_text.success("Batch processing finished.")
    return master_list

# --------------------- Streamlit Admin ---------------------
st.set_page_config(page_title="Quiz Platform", layout="wide")
st.header("📝 Admin Panel - Create Quiz")
uploaded_pdf = st.file_uploader("Upload Quiz PDF", type=["pdf"])
uploaded_csv = st.file_uploader("Upload Answer Key CSV", type=["csv"])
batch_size = st.number_input("Questions per Batch", min_value=1, max_value=50, value=10)

if st.button("Create Quiz") and uploaded_pdf and uploaded_csv:
    with st.spinner("Processing PDF and generating quiz JSON..."):
        raw_text = pdf_to_txt(uploaded_pdf)
        quiz_data = process_quiz_batches_structured(raw_text, batch_size=batch_size)

        if quiz_data:
            st.subheader("Post-Processing Report")
            for i, q in enumerate(quiz_data, 1):
                if not q['question_english']:
                    st.warning(f"⚠️ Question {i} might be incomplete.")
                for k, v in q['options'].items():
                    if not v:
                        st.warning(f"⚠️ Option '{k}' for Question {i} is empty.")

            for q in quiz_data:
                q['question_english'] = str(q.get('question_english', ''))
                q['options'] = {str(k): str(v) if v else None for k,v in q['options'].items()}

            # Read answer key
            answer_key_df = pd.read_csv(uploaded_csv)
            answer_key = {str(int(row["qno"])): str(row["answer"]).strip().upper()[:1]
                          for _, row in answer_key_df.iterrows()}

            quiz_id = str(uuid.uuid4())
            db.collection("quizzes").document(quiz_id).set({
                "title": f"Quiz {quiz_id}",
                "questions": quiz_data,
                "answer_key": answer_key,
                "created_at": firestore.SERVER_TIMESTAMP
            })
            st.success(f"✅ Quiz created successfully with ID: {quiz_id}")
        else:
            st.error("Failed to process any questions. Please check the PDF format.")
