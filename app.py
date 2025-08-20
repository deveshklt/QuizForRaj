# quiz_firestore_app_txt.py
import os
import re
import json
import uuid
import pandas as pd
import streamlit as st
from openai import OpenAI
import firebase_admin
from firebase_admin import credentials, firestore

# --------------------- Firebase Initialization ---------------------
if not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase"]))
    firebase_admin.initialize_app(cred)

db = firestore.client()

# --------------------- Utilities ---------------------
def extract_json(llm_output: str):
    cleaned = re.sub(r"^```(json)?\s*|\s*```$", "", llm_output.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)

def txt_to_str(uploaded_txt) -> str:
    return uploaded_txt.read().decode("utf-8")

def parse_raw_blocks(text: str, limit: int = 150):
    text = re.sub(r"(P\.T\.O\.|---.*?---|.*?\n)", "", text, flags=re.I)
    raw_qs = re.split(r"\n?\s*\d+\.\s+", text)
    return raw_qs[1:limit+1]

def llm_clean_questions(raw_questions, limit=150):
    prompt = f"""
You are a {limit} Questions Quiz Formatter.
Number of Questions must be exactly {limit}.
You will be given messy exam questions (Hindi + English) with options. 
There can be a question like which iss to be answered using reaading a paragraph or steps for more then one question.
Clean them and output a JSON list of objects like this:
{{
 "question_hindi": "...",
 "question_english": "...",
 "options": {{"A": "...", "B": "...", "C": "...", "D": "...", "E": "..."}}
}}
It is mandatory to create JSON of {limit} questions.
Keep bilingual versions if available. 
Ensure options are in correct A/B/C/D/E order. 
Only return JSON. 
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
    uploaded_txt = st.file_uploader("Upload Quiz TXT", type=["txt"])
    uploaded_csv = st.file_uploader("Upload Answer Key CSV", type=["csv"])
    quiz_limit = st.number_input("Number of Questions to Extract", min_value=1, max_value=150, value=10)

    if st.button("Create Quiz") and uploaded_txt and uploaded_csv:
        with st.spinner("Processing TXT and generating quiz JSON..."):
            raw_text = txt_to_str(uploaded_txt)
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

        if "last_quiz" not in st.session_state or st.session_state.last_quiz != quiz_selection:
            st.session_state.page = 0
            st.session_state.responses = {}
            st.session_state.last_quiz = quiz_selection

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

                firestore_responses = {str(k): (v[:1].upper() if v else "E")
                                       for k,v in st.session_state.responses.items()}

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
                marks = (correct*2) - (wrong*(1/3))
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
