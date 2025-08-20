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

# --------------------- Firebase Initialization ---------------------
# Load Firebase credentials from Streamlit secrets
if not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase"]))
    firebase_admin.initialize_app(cred)

db = firestore.client()

# --------------------- Utilities ---------------------
def extract_json(llm_output: str):
    cleaned = re.sub(r"^```(json)?\s*|\s*```$", "", llm_output.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)

def pdf_to_txt(uploaded_pdf) -> str:
    all_text = ""
    with pdfplumber.open(uploaded_pdf) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n\n"
    return all_text

def parse_raw_blocks(text: str, limit: int = 10):
    text = re.sub(r"(P\.T\.O\.|---.*?---|.*?\n)", "", text, flags=re.I)
    raw_qs = re.split(r"\n?\s*\d+\.\s+", text)
    return raw_qs[1:limit+1]

def llm_clean_questions(raw_questions, limit=10):
    prompt = f"""
You are a Quiz Formatter.
Format {limit} messy bilingual exam questions into JSON.
number of question should be strictly {limit}
Structure:
{{
 "question_hindi": "...",
 "question_english": "...",
 "options": {{"A":"...","B":"...","C":"...","D":"...", "E":"..."}}
}}
Return only JSON.
Questions:
{raw_questions}
"""
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0,
    )
    return extract_json(resp.choices[0].message.content.strip())

# --------------------- Streamlit Tabs ---------------------
st.set_page_config(page_title="Quiz Platform", layout="wide")
tab1, tab2 = st.tabs(["Admin Panel", "Take Quiz"])

# --------------------- ADMIN PANEL ---------------------
with tab1:
    st.header("📝 Admin Panel - Create Quiz")
    uploaded_pdf = st.file_uploader("Upload Quiz PDF", type=["pdf"])
    uploaded_csv = st.file_uploader("Upload Answer Key CSV", type=["csv"])
    quiz_limit = st.number_input("Number of Questions to Extract", min_value=1, max_value=100, value=10)

    if st.button("Create Quiz") and uploaded_pdf and uploaded_csv:
        with st.spinner("Processing PDF and generating quiz JSON..."):
            raw_text = pdf_to_txt(uploaded_pdf)
            raw_blocks = parse_raw_blocks(raw_text, limit=quiz_limit)
            quiz_data = llm_clean_questions(raw_blocks, limit=quiz_limit)

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

# --------------------- USER PANEL ---------------------
with tab2:
    st.header("🖊️ Take Quiz")
    quiz_docs = db.collection("quizzes").order_by("created_at", direction=firestore.Query.DESCENDING).stream()
    quizzes_list = [(doc.id, doc.to_dict().get("title")) for doc in quiz_docs]

    if quizzes_list:
        quiz_selection = st.selectbox("Select a Quiz", options=[q[1] for q in quizzes_list])
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
            opts = [f"{k}. {v}" for k, v in q["options"].items()]
            choice = st.radio(f"Answer for Q{idx}", options=opts, key=f"q{idx}")
            st.session_state.responses[idx] = choice

        col1, col2, col3 = st.columns([1,2,1])
        with col1:
            if st.session_state.page > 0 and st.button("⬅️ Previous"):
                st.session_state.page -= 1
                st.experimental_rerun()
        with col3:
            if st.session_state.page < total_pages - 1 and st.button("Next ➡️"):
                st.session_state.page += 1
                if hasattr(st, "rerun"):
                    st.rerun()
                    else:
                        st.experimental_rerun()

        if st.session_state.page == total_pages - 1:
            if st.button("Submit Quiz"):
                data_rows = []
                correct, wrong, unattempted = 0,0,0

                # Convert responses for Firestore: string keys + single-letter values
                firestore_responses = {str(k): (v[:1].upper() if v else "E")
                                       for k,v in st.session_state.responses.items()}

                for i in range(1, total+1):
                    sel_letter = firestore_responses.get(str(i),"E")
                    key_letter = answer_key.get(str(i),"")
                    is_correct = None

                    if sel_letter == "E":
                        unattempted += 1
                    elif sel_letter == key_letter:
                        correct +=1
                        is_correct = True
                    else:
                        wrong +=1
                        is_correct = False

                    data_rows.append({
                        "Q#": i,
                        "Selected": sel_letter,
                        "Key": key_letter,
                        "Result": "✔" if is_correct else "✘" if is_correct is False else "–"
                    })

                df = pd.DataFrame(data_rows)
                marks = (correct*2) - (wrong*(1/3))
                st.success(f"✅ Correct: {correct} | ❌ Wrong: {wrong} | 💤 Unattempted: {unattempted} | 📊 Total Marks: {marks:.2f}")
                st.dataframe(df, use_container_width=True)

                # Save sanitized responses in Firestore
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
