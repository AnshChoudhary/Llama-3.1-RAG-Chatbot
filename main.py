import ollama
import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
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

# Main chatbot function
def chatbot():
    print("Welcome to the Customer Service Chatbot!")
    print("Loading FAQ document and initializing the model...")

    # Extract text from PDF
    faq_text = extract_text_from_pdf('/Users/anshchoudhary/Downloads/llama/customer-RAG/faq.pdf')
    chunks = split_text(faq_text)

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    vectorizer.fit(chunks)

    print("Chatbot is ready. Type 'quit' to exit.")

    while True:
        user_input = input("Customer: ")
        
        if user_input.lower() == 'quit':
            print("Thank you for using our customer service. Goodbye!")
            break

        # Find most relevant chunk
        relevant_chunk = find_most_relevant_chunk(user_input, chunks, vectorizer)

        # Generate response using Llama 3.1
        prompt = f"Based on the following information from our FAQ:\n\n{relevant_chunk}\n\nCustomer question: {user_input}\n\nCustomer service response:"
        response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])

        print(f"Agent: {response['message']['content']}")

if __name__ == "__main__":
    chatbot()