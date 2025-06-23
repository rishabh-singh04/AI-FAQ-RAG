import streamlit as st
import httpx

st.title('COVID-19 FAQ Assistant')

user_input = st.text_input("Ask your COVID-19 related question here:")
if st.button("Submit"):
    response = httpx.post('http://127.0.0.1:8000/ask/', json={"question": user_input})
    answer = response.json().get('answer', "Sorry, I couldn't find an answer.")
    st.text("Answer: " + answer)