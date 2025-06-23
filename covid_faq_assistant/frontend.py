import streamlit as st
import requests

st.title('CoviBuddy : Your AI Assistant')

user_question = st.text_input("Ask a question:")

if st.button("Submit"):
    try:
        response = requests.post("http://localhost:8000/ask/", json={"question": user_question})
        if response.ok:
            st.write(response.json())
        else:
            st.error(f"Failed to retrieve data, status code: {response.status_code}")
    except requests.exceptions.ConnectionError as e:
        st.error("Failed to connect to the backend. Please ensure the backend service is running.")