import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("📄 AI-Powered Resume Analyzer")

# Upload CSV
uploaded_file = st.file_uploader("📥 Upload CSV Resume Dataset", type=["csv"])
job_description = st.text_area("📝 Enter the Job Description", height=200)

if uploaded_file and job_description:
    # Load resumes
    try:
        df = pd.read_csv(uploaded_file)
        if 'Resume' not in df.columns:
            st.error("CSV must have a 'Resume' column.")
        else:
            # TF-IDF Vectorization
            resumes = df['Resume'].astype(str)
            documents = resumes.tolist() + [job_description]
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(documents)

            # Cosine Similarity
            job_vector = tfidf_matrix[-1]
            resume_vectors = tfidf_matrix[:-1]
            similarities = cosine_similarity(job_vector, resume_vectors)[0]

            # Add scores to DataFrame
            df['Match Score (%)'] = (similarities * 100).round(2)
            df_sorted = df.sort_values(by='Match Score (%)', ascending=False)

            st.success("✅ Resumes analyzed successfully!")
            st.dataframe(df_sorted[['Resume', 'Match Score (%)']])
    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.warning("Please upload a CSV file and enter a job description to get started.")
