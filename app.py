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
    # Normalize the text by ensuring a newline before each question number
    normalized_text = re.sub(r'(\d+)\.', r'\n\1.', text).strip()
    
    # Split the text into individual questions based on the normalized numbers
    questions = re.split(r'\n(\d+\.)', normalized_text)
    
    # Filter and re-join the questions and their numbers
    questions_list = []
    for i in range(1, len(questions), 2):
        questions_list.append(questions[i] + questions[i+1])
    
    return questions_list

def llm_clean_questions_streaming(raw_questions: str):
    """
    Sends a batch of raw questions to an LLM and streams the response.
    
    Args:
        raw_questions (str): A string containing the messy questions.
        
    Yields:
        str: Chunks of the LLM's response as they are generated.
    """
    prompt = f"""
You are a Quiz Formatter.
You will be given messy exam questions (Hindi + English) with options. 
Clean them and output a JSON list of objects like this:
{{
 "question_hindi": "...",
 "question_english": "...",
 "options": {{"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."}}
}}
Create a JSON list of the questions provided.
Keep bilingual versions if available. 
Ensure options are in correct A/B/C/D/E order. 
Only return JSON. DO NOT use any markdown formatting, code fences, or additional text.
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

# --------------------- ADMIN PANEL ---------------------
with tab1:
    st.header("📝 Admin Panel - Create Quiz")
    uploaded_pdf = st.file_uploader("Upload Quiz PDF", type=["pdf"])
    uploaded_csv = st.file_uploader("Upload Answer Key CSV", type=["csv"])
    batch_size = st.number_input("Questions per Batch", min_value=1, max_value=50, value=10)
    
    if st.button("Create Quiz") and uploaded_pdf and uploaded_csv:
        with st.spinner("Processing PDF and generating quiz JSON..."):
            raw_text = pdf_to_txt(uploaded_pdf)
            quiz_data = process_quiz_batches(raw_text, batch_size=batch_size)

            if quiz_data:
                # Ensure all quiz fields are strings
                for q in quiz_data:
                    q['question_hindi'] = str(q.get('question_hindi', ''))
                    q['question_english'] = str(q.get('question_english', ''))
                    q['options'] = {str(k): str(v) for k,v in q['options'].items()}

                # Read answer key CSV into dict with string keys
                answer_key_df = pd.read_csv(uploaded_csv)
                answer_key = {str(int(row["qno"])): str(row["answer"]).strip().upper()[:1]
                              for _, row in answer_key_df.iterrows()}

                # Save to Firestore
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

# --------------------- USER PANEL ---------------------
with tab2:
    st.header("🖊️ Take Quiz")
    quiz_docs = db.collection("quizzes").order_by("created_at", direction=firestore.Query.DESCENDING).stream()
    quizzes_list = [(doc.id, doc.to_dict().get("title")) for doc in quiz_docs]

    if quizzes_list:
        quiz_selection = st.selectbox("Select a Quiz", options=[q[1] for q in quizzes_list])

        # ✅ Reset page if quiz changes
        if "last_quiz" not in st.session_state or st.session_state.last_quiz != quiz_selection:
            st.session_state.page = 0
            st.session_state.responses = {}
            st.session_state.last_quiz = quiz_selection
            st.rerun()

        quiz_id = [q[0] for q in quizzes_list if q[1] == quiz_selection][0]

        quiz_doc = db.collection("quizzes").document(quiz_id).get().to_dict()
        quiz_data = quiz_doc["questions"]
        answer_key = quiz_doc["answer_key"]

        if "responses" not in st.session_state:
            st.session_state.responses = {}
        if "page" not in st.session_state:
            st.session_state.page = 0

        per_page = 5
        total = len(quiz_data)
        total_pages = (total - 1) // per_page + 1
        start = st.session_state.page * per_page
        end = start + per_page
        current_questions = quiz_data[start:end]

        for idx, q in enumerate(current_questions, start=start+1):
            st.markdown(f"**Q{idx}. {q['question_hindi']}**")
            if q.get("question_english"):
                st.caption(q["question_english"])
            
            # Find the selected option letter for the current question
            selected_option = st.session_state.responses.get(idx, None)
            
            # Map options to their display text
            options_dict = q.get('options', {})
            display_options = [f"{k}. {v}" for k, v in options_dict.items()]
            
            # Find the index of the previously selected option
            if selected_option:
                try:
                    selected_index = display_options.index(selected_option)
                except ValueError:
                    selected_index = None # Or handle the case where the option is not found
            else:
                selected_index = None
            
            choice = st.radio(
                f"Answer for Q{idx}", 
                options=display_options, 
                key=f"q{idx}",
                index=selected_index
            )
            st.session_state.responses[idx] = choice

        # ---------------- Page Navigation ----------------
        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            if st.session_state.page > 0 and st.button("⬅️ Previous"):
                st.session_state.page -= 1
                st.rerun()
        with col3:
            if st.session_state.page < total_pages - 1 and st.button("Next ➡️"):
                st.session_state.page += 1
                st.rerun()

        # ✅ Page numbers at bottom
        with col2:
            page_buttons = st.columns(total_pages)
            for i in range(total_pages):
                if page_buttons[i].button(str(i+1), key=f"page{i}"):
                    st.session_state.page = i
                    st.rerun()

        # ---------------- Submit Section ----------------
        if st.session_state.page == total_pages - 1:
            if st.button("Submit Quiz"):
                data_rows = []
                correct, wrong, unattempted = 0, 0, 0

                firestore_responses = {}
                for q_num, choice_str in st.session_state.responses.items():
                    if choice_str:
                        # Extract the letter (A, B, C, D, E) from the full string
                        letter = choice_str.split('.')[0].strip()
                        firestore_responses[str(q_num)] = letter.upper()
                    else:
                        firestore_responses[str(q_num)] = "E"

                for i in range(1, total+1):
                    sel_letter = firestore_responses.get(str(i),"E")
                    key_letter = answer_key.get(str(i),"")
                    is_correct = None

                    if sel_letter == "E":
                        unattempted += 1
                    elif sel_letter == key_letter:
                        correct += 1
                        is_correct = True
                    else:
                        wrong += 1
                        is_correct = False

                    data_rows.append({
                        "Q#": i,
                        "Selected": sel_letter,
                        "Key": key_letter,
                        "Result": "✔" if is_correct else "✘" if is_correct is False else "–"
                    })

                df = pd.DataFrame(data_rows)
                marks = (correct * 2) - (wrong * (1/3))
                st.success(f"✅ Correct: {correct} | ❌ Wrong: {wrong} | 💤 Unattempted: {unattempted} | 📊 Total Marks: {marks:.2f}")
                st.dataframe(df, use_container_width=True)

                user_id = str(uuid.uuid4())
                db.collection("responses").document(f"{quiz_id}_{user_id}").set({
                    "quiz_id": str(quiz_id),
                    "user_id": str(user_id),
                    "responses": firestore_responses,
                    "submitted_at": firestore.SERVER_TIMESTAMP
                })
                st.info("Responses saved successfully.")
    else:
        st.info("No quizzes available yet. Admin needs to create one.")
