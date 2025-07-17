import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("UpdatedResumeDataSet.csv")

# Show first few resumes
print("Sample resumes:")
print(df[['Category', 'Resume']].head())

# Define a sample job description to compare
job_description = """
We are looking for a Data Scientist with experience in Python, machine learning, and data analysis.
Familiarity with pandas, scikit-learn, and statistics is required.
"""

# Use count vectorizer to match resumes with job description
cv = CountVectorizer(stop_words='english')
vectors = cv.fit_transform([job_description] + list(df['Resume'][:5]))  # Compare first 5 resumes

# Cosine similarity
similarity_scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

# Print scores
print("\nJob match scores for first 5 resumes:")
for i, score in enumerate(similarity_scores):
    print(f"Resume {i+1} match score: {round(score * 100, 2)}%")
