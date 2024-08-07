import streamlit as st
from streamlit_chat import message
import ollama
import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config for dark theme
st.set_page_config(page_title="Chat with Llama 3.1", layout="centered", initial_sidebar_state="collapsed")

# Apply custom CSS for dark theme
st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stTextInput > div > div > input {
        background-color: #2B2B2B;
        color: #FFFFFF;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)

# Rest of your functions (extract_text_from_pdf, split_text, find_most_relevant_chunk) remain the same
# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split text into chunks
def split_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Function to find most relevant chunk
def find_most_relevant_chunk(query, chunks, vectorizer):
    query_vector = vectorizer.transform([query])
    chunk_vectors = vectorizer.transform(chunks)
    similarities = cosine_similarity(query_vector, chunk_vectors)
    most_relevant_idx = np.argmax(similarities)
    return chunks[most_relevant_idx]

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'faq_loaded' not in st.session_state:
    st.session_state.faq_loaded = False

# Streamlit app
st.title("Bella Luna Restaurant - Customer Support")

# File uploader for FAQ PDF
if not st.session_state.faq_loaded:
    uploaded_file = st.file_uploader("Upload FAQ PDF", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing FAQ document..."):
            faq_text = extract_text_from_pdf(uploaded_file)
            chunks = split_text(faq_text)
            vectorizer = TfidfVectorizer()
            vectorizer.fit(chunks)
            st.session_state.chunks = chunks
            st.session_state.vectorizer = vectorizer
            st.session_state.faq_loaded = True
        st.success("FAQ document loaded successfully!")
        st.experimental_rerun()

# Chat interface
if st.session_state.faq_loaded:
    # Display chat messages
    for i, chat in enumerate(st.session_state.messages):
        message(chat["content"], is_user=chat["role"] == "user", key=str(i))
    
    # Chat input
    user_input = st.text_input("What is your message?", key="user_input")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Find most relevant chunk
        relevant_chunk = find_most_relevant_chunk(user_input, st.session_state.chunks, st.session_state.vectorizer)

        # Generate response using Llama 3.1
        prompt = f"You are a friendly and helpful AI assistant. Based on the following information from our FAQ:\n\n{relevant_chunk}\n\nCustomer question: {user_input}\n\nRespond in a conversational manner:"
        with st.spinner("Thinking..."):
            response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])

        ai_response = response['message']['content']
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        
        # Force a rerun to update the chat display
        st.experimental_rerun()

else:
    st.info("Please upload an FAQ document to start the chat.")

