import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Analyzer", layout="centered")

st.title("📄 AI-Powered Resume Analyzer")
st.write("Upload your resume dataset (CSV) and a job description to find the best-matched resumes.")

# File uploader
uploaded_file = st.file_uploader("📂 Upload Resume Dataset (CSV)", type=["csv"])

# Job Description input
job_description = st.text_area("📝 Enter the Job Description", height=200)

# When both file and job description are provided
if uploaded_file is not None and job_description:
    # Load resumes
    df = pd.read_csv(uploaded_file)

    if 'Resume' not in df.columns:
        st.error("❌ CSV must have a 'Resume' column.")
    else:
        # Vectorize resumes and job description
        vectorizer = TfidfVectorizer(stop_words='english')
        resume_vectors = vectorizer.fit_transform(df['Resume'].astype(str))
        job_vector = vectorizer.transform([job_description])

        # Calculate similarity
        similarity_scores = cosine_similarity(job_vector, resume_vectors).flatten()
        df['Match Score (%)'] = (similarity_scores * 100).round(2)

        # Sort by best match
        top_matches = df.sort_values(by='Match Score (%)', ascending=False)

        # Display top 5 matches
        st.subheader("🔍 Top Matching Resumes:")
        st.dataframe(top_matches[['Resume', 'Match Score (%)']].head(5))

        # Option to download results
        st.download_button("⬇️ Download Full Match Results as CSV", top_matches.to_csv(index=False), file_name='resume_matches.csv', mime='text/csv')
else:
    st.info("Please upload a CSV file and provide a job description to get started.")
