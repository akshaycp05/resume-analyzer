# interview_chatbot.py
import openai
import streamlit as st

# Load API key securely
openai.api_key = st.secrets["openai_api_key"]

st.title("🤖 AI Interview Coach")
st.markdown("Ask questions or practice your interview based on your resume!")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are an AI Interview Coach. Ask or answer questions related to resume skills and job descriptions. Be professional."}
    ]

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask me an interview question or describe a role...")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state["messages"]
        )
        reply = response.choices[0].message.content
        st.markdown(reply)

    st.session_state["messages"].append({"role": "assistant", "content": reply})
