# interview_chatbot.py

import openai
import streamlit as st

def generate_interview_question(resume_text, job_description):
    prompt = f"""
    You are a professional HR Interviewer. Based on the following resume and job description, ask a relevant interview question.
    
    Resume: {resume_text}
    Job Description: {job_description}
    
    Interview Question:
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )

    return response['choices'][0]['message']['content'].strip()


def chat_with_candidate(user_input, resume_text, job_description):
    prompt = f"""
    You are simulating an AI Interview Coach. Based on the candidate's response, ask a follow-up question or provide feedback. Stay professional.

    Resume: {resume_text}
    Job Description: {job_description}
    Candidate Response: {user_input}
    
    AI Interviewer:
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.7
    )

    return response['choices'][0]['message']['content'].strip()
