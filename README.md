# Llama 3.1 RAG Chatbot

This project implements a conversational AI chatbot using the Llama 3.1 language model and a Retrieval-Augmented Generation (RAG) approach. The chatbot is designed to answer questions based on a provided FAQ document, combining the power of a large language model with specific domain knowledge.

## Features

- Web-based chat interface using Streamlit
- PDF document upload for FAQ/knowledge base
- RAG (Retrieval-Augmented Generation) for context-aware responses
- Integration with Llama 3.1 language model via Ollama
- Dark mode UI for better user experience

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/llama-rag-chatbot.git
cd llama-rag-chatbot
```
2. Install the required dependencies:
```bash
pip install streamlit streamlit-chat ollama PyPDF2 numpy scikit-learn
```

3. Ensure you have Ollama installed and the Llama 3.1 model downloaded:
```bash
ollama pull llama:3.1
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

4. Upload your FAQ PDF document when prompted.

5. Start chatting with the AI assistant!

## How It Works

### RAG (Retrieval-Augmented Generation) Approach

This chatbot uses a RAG approach to provide informed responses:

1. **Document Ingestion**: The uploaded PDF is processed and its text is extracted.

2. **Text Chunking**: The extracted text is split into manageable chunks.

3. **Vectorization**: TF-IDF vectorization is applied to create numerical representations of the text chunks.

4. **Retrieval**: When a user asks a question, the system finds the most relevant chunk of text using cosine similarity between the question and the text chunks.

5. **Generation**: The retrieved chunk, along with the user's question, is sent to the Llama 3.1 model to generate a contextually relevant response.

This approach allows the chatbot to leverage both the broad knowledge of the language model and the specific information from the provided document.

### Llama 3.1 Integration

Llama 3.1 is a powerful open-source language model developed by Meta AI. In this project, we use Ollama to run Llama 3.1 locally. Key aspects of the integration:

- **Model Access**: Ollama provides a simple API to interact with Llama 3.1.
- **Prompt Engineering**: We construct a prompt that includes the retrieved context and the user's question, instructing the model to respond in a conversational manner.
- **Response Generation**: Llama 3.1 generates human-like responses based on the provided context and question.

The combination of RAG and Llama 3.1 allows for responses that are both informative (based on the FAQ) and natural-sounding.

## Customization

- Modify the `app.py` file to adjust the UI or add new features.
- Experiment with different chunking sizes or vectorization methods in the RAG approach.
- Try different prompts or model parameters to fine-tune the Llama 3.1 responses.

## Contributing

Contributions to improve the chatbot are welcome! Please feel free to submit pull requests or open issues to discuss potential enhancements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
