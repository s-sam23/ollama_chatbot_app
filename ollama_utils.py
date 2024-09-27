import streamlit as st
import subprocess

# Function to start the Ollama server asynchronously
def start_ollama_server():
    try:
        st.write("Starting the Ollama server...")
        subprocess.Popen(["ollama", "serve"])
        st.success("Ollama server started successfully.")
    except Exception as e:
        st.error(f"Error starting Ollama server: {e}")

# Function to pull a model
def pull_model(model_name):
    try:
        st.write(f"Pulling the model: {model_name}...")
        subprocess.run(["ollama", "pull", model_name], check=True)
        st.success(f"Model '{model_name}' pulled successfully.")
    except subprocess.CalledProcessError as e:
        st.error(f"Error pulling the model: {e}")
