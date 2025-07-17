import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page title
st.set_page_config(page_title="Smart Resume Analyzer", layout="centered")
st.title("📄 AI-Powered Resume Analyzer")
st.write("Upload your resume dataset and enter a job description to see which resumes match best.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV Resume Dataset", type="csv")

# Text input for job description
job_desc = st.text_area("📝 Enter the Job Description", height=200)

if uploaded_file is not None and job_desc.strip() != "":
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Check for required columns
    if 'Resume' not in df.columns or 'Category' not in df.columns:
        st.error("CSV file must contain 'Resume' and 'Category' columns.")
    else:
        # Vectorize resumes and job description
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df['Resume'].values.astype('U'))

        # Transform job description
        job_vector = vectorizer.transform([job_desc])

        # Compute cosine similarity
        similarity_scores = cosine_similarity(tfidf_matrix, job_vector).flatten()

        # Add match percentage
        df['Match %'] = (similarity_scores * 100).round(2)

        # Skill matching
        keywords = [word.lower() for word in job_desc.split() if len(word) > 3]

        def extract_skills(resume):
            resume_words = resume.lower().split()
            matched = [kw for kw in keywords if kw in resume_words]
            return ", ".join(set(matched))

        df['Matched Skills'] = df['Resume'].apply(extract_skills)

        # Tips based on match percentage
        def give_tip(score):
            if score >= 80:
                return "✅ Excellent match! Ready to apply."
            elif score >= 60:
                return "⚠️ Improve by including more job keywords."
            else:
                return "❌ Resume needs significant improvement."

        df['Tip'] = df['Match %'].apply(give_tip)

        # Show results
        st.subheader("📊 Resume Matching Results")
        st.dataframe(df[['Category', 'Match %', 'Matched Skills', 'Tip']].sort_values(by='Match %', ascending=False).reset_index(drop=True))

else:
    st.info("Please upload a CSV file and provide a job description to get started.")
