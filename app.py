# app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from interview_chatbot import generate_interview_question, chat_with_candidate
import base64

st.set_page_config(page_title="AI-Powered Resume Analyzer", layout="centered")
st.title("📄 AI-Powered Resume Analyzer")

st.markdown("""
Upload your **CSV resume dataset** and enter a **job description**.
The AI will analyze which resumes are the best match and you can launch an **AI Interview Coach** for practice.
""")

uploaded_file = st.file_uploader("📤 Upload CSV Resume Dataset", type=["csv"])
job_description = st.text_area("📝 Enter the Job Description")

if uploaded_file and job_description:
    df = pd.read_csv(uploaded_file)
    if 'Resume' not in df.columns:
        st.error("CSV must contain a column named 'Resume'.")
    else:
        st.success("✅ File and job description accepted!")

        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(df['Resume'].astype(str))
        job_vec = vectorizer.transform([job_description])

        similarities = cosine_similarity(job_vec, tfidf_matrix).flatten()
        df['Score'] = similarities
        top_matches = df.sort_values(by='Score', ascending=False).head(5)

        st.subheader("🏆 Top Matching Resumes")
        st.dataframe(top_matches[['Resume', 'Score']])

        st.download_button(
            label="⬇ Download Top 5 Matches as CSV",
            data=top_matches.to_csv(index=False).encode('utf-8'),
            file_name='top_matches.csv',
            mime='text/csv'
        )

        # --- Interview Coach ---
        if st.checkbox("🧠 Launch AI Interview Coach"):
            st.subheader("💬 Mock Interview Chatbot")

            resume_text = st.text_area("📌 Paste a selected resume")
            job_text = st.text_area("📌 Paste the job description again")

            if st.button("🎯 Generate First Interview Question"):
                if resume_text and job_text:
                    question = generate_interview_question(resume_text, job_text)
                    st.session_state.chat_history = [
                        {"role": "system", "content": "You are an AI Interviewer. Ask job-related questions."},
                        {"role": "assistant", "content": question}
                    ]
                else:
                    st.warning("Please provide both resume and job description.")

            if "chat_history" in st.session_state:
                for msg in st.session_state.chat_history[1:]:
                    st.chat_message(msg["role"]).write(msg["content"])

                user_input = st.chat_input("Type your response here")
                if user_input:
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    ai_reply = chat_with_candidate(st.session_state.chat_history)
                    st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
                    st.experimental_rerun()

else:
    st.warning("Please upload a CSV file and provide a job description to get started.")
