import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Analyzer & Interview Coach", layout="centered")

st.title("📄 AI-Powered Resume Analyzer")
st.markdown("Upload your resume dataset and enter a job description to see which resumes match best and get interview questions.")

# Step 1: Upload CSV
uploaded_file = st.file_uploader("📤 Upload CSV Resume Dataset", type=["csv"])
job_description = st.text_area("📝 Enter the Job Description")

if uploaded_file and job_description:
    data = pd.read_csv(uploaded_file)

    if "Resume" not in data.columns:
        st.error("❌ CSV must contain a 'Resume' column.")
    else:
        st.success("✅ Data loaded successfully!")

        # Step 2: Vectorize
        tfidf = TfidfVectorizer(stop_words="english")
        resume_tfidf = tfidf.fit_transform(data["Resume"].astype(str))
        job_tfidf = tfidf.transform([job_description])

        # Step 3: Similarity
        similarity_scores = cosine_similarity(job_tfidf, resume_tfidf).flatten()
        data["Match_Score"] = similarity_scores
        top_matches = data.sort_values("Match_Score", ascending=False).head(5)

        st.markdown("### 🏆 Top 5 Matching Resumes")
        st.dataframe(top_matches[["Resume", "Match_Score"]], use_container_width=True)

        # Interview Coach
        if st.button("🎯 Show Interview Questions for Top Match"):
            top_resume = top_matches.iloc[0]["Resume"]

            # Step 4: Extract Skills
            common_skills = [
                "Python", "Machine Learning", "Data Science", "SQL", "Deep Learning",
                "Java", "Communication", "Teamwork", "Leadership", "NLP",
                "Computer Vision", "Project Management"
            ]
            found_skills = [skill for skill in common_skills if skill.lower() in top_resume.lower()]

            st.markdown("### 🤖 Top Skills Detected:")
            if found_skills:
                st.write(", ".join(found_skills))
            else:
                st.write("No common skills found.")

            st.markdown("### 🧪 Sample Interview Questions:")
            for skill in found_skills:
                st.write(f"**{skill}**:")
                if skill == "Python":
                    st.write("- What are Python decorators?")
                    st.write("- How does Python manage memory?")
                elif skill == "Machine Learning":
                    st.write("- What is overfitting and how do you prevent it?")
                    st.write("- Explain supervised vs unsupervised learning.")
                elif skill == "SQL":
                    st.write("- What is a JOIN? Different types?")
                    st.write("- Write a SQL query to find duplicates.")
                elif skill == "Deep Learning":
                    st.write("- What is backpropagation?")
                    st.write("- Difference between CNN and RNN?")
                elif skill == "Java":
                    st.write("- Explain inheritance in Java.")
                    st.write("- What is the JVM?")
                elif skill == "Communication":
                    st.write("- Describe a time you resolved a conflict.")
                elif skill == "Leadership":
                    st.write("- How do you motivate a team?")
                elif skill == "Project Management":
                    st.write("- Tools used in project management?")
                    st.write("- Agile vs Waterfall?")
                else:
                    st.write(f"- Tell us about your experience with {skill}.")

            st.info("✅ Interview questions generated based on resume content.")
else:
    st.info("Please upload a CSV file and enter a job description to begin.")
