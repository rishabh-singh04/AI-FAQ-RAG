import streamlit as st
import requests

st.set_page_config(page_title="COVID-19 FAQ Assistant", layout="wide")

API_URL = "http://127.0.0.1:8000/get_faq_response"  # FastAPI endpoint

# --- Streamlit UI Layout ---
st.title("CoviBuddy: Your AI Assistant")
st.write("Ask any question related to COVID-19, and our AI will provide information.")

# --- Initialize Conversation Context ---
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# --- Chat Input Box ---
user_query = st.text_input("Enter your question:", "")

# --- Submit Button ---
if st.button("Get Answer") and user_query:
    with st.spinner("Retrieving your answer..."):
        payload = {"query": user_query, "history": st.session_state.conversation}

        try:
            response = requests.post(API_URL, json=payload)
            response_data = response.json()

            if response.status_code == 200:
                enhanced_response = response_data["enhanced_response"]
                st.session_state.conversation.append({"question": user_query, "answer": enhanced_response})
                
                st.subheader("🤖 AI-enhanced Response:")
                st.success(enhanced_response)
            else:
                st.error("No relevant FAQ found. Try rephrasing your query.")
        
        except Exception as e:
            st.error(f"Error fetching response: {e}")

# --- Display Conversation History ---
if st.session_state.conversation:
    st.markdown("### Conversation History")
    for i, convo in enumerate(st.session_state.conversation, 1):
        st.write(f"**Q{i}:** {convo['question']}")
        st.write(f"**A{i}:** {convo['answer']}")

# --- About Section ---
st.markdown("---")
#st.markdown("💡 **Powered by FAISS, OpenAI (DIAL API), and Streamlit**")

 