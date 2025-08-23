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
                # Post-processing and logging of potential issues
                st.subheader("Post-Processing Report")
                
                for i, q in enumerate(quiz_data, 1):
                    # Check for null values indicating parsing issues
                    if not q.get('question_english'):
                        st.warning(f"⚠️ Question {i} might be incomplete.")
                    for k, v in q['options'].items():
                        if not v:
                            st.warning(f"⚠️ Option '{k}' for Question {i} is empty.")
                
                # Ensure all quiz fields are strings
                for q in quiz_data:
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
            st.markdown(f"**Q{idx}. {q['question_english']}**")
            
            # Find the selected option letter for the current question
            selected_option = st.session_state.responses.get(idx, None)
            
            # Map options to their display text
            options_dict = q.get('options', {})
            display_options = [f"{k}. {v}" for k, v in options_dict.items() if v] # Filter out null options
            
            # Find the index of the previously selected option
            if selected_option:
                try:
                    selected_index = display_options.index(selected_option)
                except ValueError:
                    selected_index = None
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
