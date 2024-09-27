import streamlit as st
import subprocess
from chatbot import chat_with_ollama  # Import the chatbot logic
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from config import DATA_PATH, CHROMA_PATH
from langchain.vectorstores.chroma import Chroma
import ollama


# Automatically start the Ollama server when the app runs
def start_ollama_server():
    try:
        # Start Ollama server in the background without user interaction
        subprocess.Popen(["ollama", "serve"])  # Run the server in the background
    except Exception as e:
        st.error(f"An error occurred while starting the server: {e}")

# Automatically pull the specified model
def pull_model(model_name="qwen2.5:0.5b"):
    try:
        # Pull the model silently without user interaction
        subprocess.run(["ollama", "pull", model_name], check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"An error occurred while pulling the model: {e}")
        
def load_existing_chroma_db(model_name="qwen2.5:0.5b"):
    # Initialize the embedding function with the Ollama model
    embedding_function = OllamaEmbeddings(model=model_name)
    
    # Load the existing ChromaDB using the persist_directory path
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    return db

# Automatically start the server and pull the model when the app is run
if 'server_started' not in st.session_state:
    start_ollama_server()
    pull_model()
    st.session_state['server_started'] = True
    
model = Ollama(model="qwen2.5:0.5b")
# Define the path where your existing ChromaDB is stored
# CHROMA_PATH = 'vector_db/chroma'

# Function to load the existing ChromaDB
db = load_existing_chroma_db()


st.set_page_config(page_title="🤗💬 HugChat")

with st.sidebar:
    st.title('🤗💬 Buzz_Board_Bot')
    st.image('img/BB-Logo.jpg')
    st.markdown('📖 Learn how to build this app in this [site](https://workable.com/nr?l=http%3A%2F%2Fwww.buzzboard.com%2F)!')
    
    
    
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            response = chat_with_ollama(db,prompt,model) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
